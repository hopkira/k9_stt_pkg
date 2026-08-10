#include <algorithm>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cctype>
#include <cstdint>
#include <cstdio>
#include <deque>
#include <functional>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

#include "rclcpp/rclcpp.hpp"
#include "std_msgs/msg/string.hpp"
#include "k9_interfaces_pkg/msg/audio_frame.hpp"
#include "whisper.h"

namespace {
constexpr const char * LISTENING = "LISTENING";
constexpr int SAMPLE_RATE = 16000;
constexpr std::size_t VAD_FRAME_SAMPLES = 512; // Silero v6 @ 16 kHz

std::string normalise(std::string s) {
  while (!s.empty() && std::isspace(static_cast<unsigned char>(s.front()))) s.erase(s.begin());
  while (!s.empty() && std::isspace(static_cast<unsigned char>(s.back()))) s.pop_back();
  std::transform(s.begin(), s.end(), s.begin(), [](unsigned char c) {
    return c == ' ' ? '_' : static_cast<char>(std::toupper(c));
  });
  return s;
}

std::string trim(std::string s) {
  while (!s.empty() && std::isspace(static_cast<unsigned char>(s.front()))) s.erase(s.begin());
  while (!s.empty() && std::isspace(static_cast<unsigned char>(s.back()))) s.pop_back();
  return s;
}

bool useful_text(const std::string & s) {
  if (s.size() < 2) return false;
  return std::any_of(s.begin(), s.end(), [](unsigned char c) { return std::isalpha(c) != 0; });
}
} // namespace

class SpeechToTextNode : public rclcpp::Node {
public:
  SpeechToTextNode() : Node("k9_stt") {
    audio_topic_ = declare_parameter<std::string>("audio_topic", "/audio/raw");
    effective_state_topic_ = declare_parameter<std::string>(
      "effective_state_topic", "/audio/effective_state");
    text_topic_ = declare_parameter<std::string>("text_topic", "/speech_to_text/text");
    diagnostic_topic_ = declare_parameter<std::string>("state_topic", "/speech_to_text/state");

    sample_rate_ = declare_parameter<int>("sample_rate", SAMPLE_RATE);
    silence_timeout_ = declare_parameter<double>("silence_timeout", 1.0);
    max_speech_s_ = declare_parameter<double>("max_speech_s", 30.0);
    pre_roll_ms_ = declare_parameter<int>("pre_roll_ms", 300);
    min_speech_ms_ = declare_parameter<int>("min_speech_ms", 160);

    model_path_ = declare_parameter<std::string>(
      "model_path", "/home/hopkira/whisper.cpp/models/ggml-large-v3-turbo.bin");
    use_gpu_ = declare_parameter<bool>("use_gpu", true);
    gpu_device_ = declare_parameter<int>("gpu_device", 0);
    flash_attn_ = declare_parameter<bool>("flash_attn", false);
    whisper_threads_ = declare_parameter<int>("whisper_threads", 4);
    beam_size_ = declare_parameter<int>("beam_size", 5);
    language_ = declare_parameter<std::string>("language", "en");
    publish_stale_results_ = declare_parameter<bool>("publish_stale_results", false);

    vad_model_path_ = declare_parameter<std::string>(
      "vad_model_path", "/home/hopkira/whisper.cpp/models/ggml-silero-v6.2.0.bin");
    vad_threads_ = declare_parameter<int>("vad_threads", 2);

    if (sample_rate_ != SAMPLE_RATE) {
      throw std::runtime_error("k9_stt requires 16000 Hz audio");
    }
    if (silence_timeout_ <= 0.0) {
      throw std::runtime_error("silence_timeout must be greater than zero");
    }
    if (max_speech_s_ <= 0.0) {
      throw std::runtime_error("max_speech_s must be greater than zero");
    }

    pre_roll_max_samples_ = static_cast<std::size_t>(
      sample_rate_ * std::max(0, pre_roll_ms_) / 1000);
    min_voiced_samples_ = static_cast<std::size_t>(
      sample_rate_ * std::max(0, min_speech_ms_) / 1000);
    silence_timeout_samples_ = static_cast<std::size_t>(
      static_cast<double>(sample_rate_) * silence_timeout_);
    max_speech_samples_ = static_cast<std::size_t>(
      static_cast<double>(sample_rate_) * max_speech_s_);

    auto cparams = whisper_context_default_params();
    cparams.use_gpu = use_gpu_;
    cparams.gpu_device = gpu_device_;
    cparams.flash_attn = flash_attn_;

    RCLCPP_INFO(
      get_logger(), "Whisper system: %s", whisper_print_system_info());
    RCLCPP_INFO(
      get_logger(), "Loading Whisper model %s; GPU=%s device=%d flash_attn=%s",
      model_path_.c_str(), use_gpu_ ? "true" : "false", gpu_device_,
      flash_attn_ ? "true" : "false");

    whisper_ctx_ = whisper_init_from_file_with_params(model_path_.c_str(), cparams);
    if (!whisper_ctx_) {
      throw std::runtime_error("Failed to load Whisper model: " + model_path_);
    }

    auto vparams = whisper_vad_default_context_params();
    vparams.n_threads = vad_threads_;
    vparams.use_gpu = false; // tiny VAD on CPU; reserve GPU for Whisper
    vparams.gpu_device = 0;

    RCLCPP_INFO(get_logger(), "Loading Silero VAD %s", vad_model_path_.c_str());
    vad_ctx_ = whisper_vad_init_from_file_with_params(vad_model_path_.c_str(), vparams);
    if (!vad_ctx_) {
      whisper_free(whisper_ctx_);
      whisper_ctx_ = nullptr;
      throw std::runtime_error("Failed to load VAD model: " + vad_model_path_);
    }

    text_pub_ = create_publisher<std_msgs::msg::String>(text_topic_, 10);
    diagnostic_pub_ = create_publisher<std_msgs::msg::String>(diagnostic_topic_, 10);

    rclcpp::QoS audio_qos(rclcpp::KeepLast(10));
    audio_qos.best_effort().durability_volatile();
    audio_sub_ = create_subscription<k9_interfaces_pkg::msg::AudioFrame>(
      audio_topic_, audio_qos,
      std::bind(&SpeechToTextNode::audio_callback, this, std::placeholders::_1));
    state_sub_ = create_subscription<std_msgs::msg::String>(
      effective_state_topic_, 10,
      std::bind(&SpeechToTextNode::state_callback, this, std::placeholders::_1));

    worker_ = std::thread(&SpeechToTextNode::worker_loop, this);

    publish_state("inactive");
    RCLCPP_INFO(
      get_logger(),
      "K9 whisper.cpp GPU STT ready; whisper.cpp=%s, silence=%.2fs, max_speech=%.2fs, "
      "pre_roll=%dms, min_speech=%dms",
      whisper_version(), silence_timeout_, max_speech_s_, pre_roll_ms_, min_speech_ms_);
  }

  ~SpeechToTextNode() override {
    stop_.store(true);
    work_cv_.notify_all();
    if (worker_.joinable()) worker_.join();
    if (vad_ctx_) whisper_vad_free(vad_ctx_);
    if (whisper_ctx_) whisper_free(whisper_ctx_);
  }

private:
  struct WorkItem {
    std::uint64_t session;
    std::vector<float> audio;
  };

  void state_callback(const std_msgs::msg::String::SharedPtr msg) {
    const auto new_state = normalise(msg->data);
    if (new_state == effective_state_) return;

    const auto old_state = effective_state_;
    effective_state_ = new_state;

    if (new_state == LISTENING) {
      const auto s = session_.fetch_add(1) + 1;

      // A new listening session makes queued work from an earlier session obsolete
      // unless stale-result publication was explicitly requested.
      if (!publish_stale_results_) {
        purge_pending_work("new LISTENING session");
      }

      listening_.store(true);
      latched_.store(false);
      reset_capture();
      whisper_vad_reset_state(vad_ctx_);
      publish_state("listening");

      RCLCPP_INFO(
        get_logger(), "%s -> LISTENING; STT session %lu",
        old_state.empty() ? "<unset>" : old_state.c_str(),
        static_cast<unsigned long>(s));
    } else {
      listening_.store(false);
      latched_.store(true);

      // Prevent stale queued audio from being transcribed after STT has been disabled
      // unless stale-result publication was explicitly requested.
      if (!publish_stale_results_) {
        purge_pending_work("left LISTENING");
      }

      reset_capture();
      whisper_vad_reset_state(vad_ctx_);
      publish_state("inactive");

      RCLCPP_INFO(
        get_logger(), "%s -> %s; STT disabled",
        old_state.empty() ? "<unset>" : old_state.c_str(), new_state.c_str());
    }
  }

  void audio_callback(const k9_interfaces_pkg::msg::AudioFrame::SharedPtr msg) {
    if (!listening_.load() || latched_.load()) return;

    if (static_cast<int>(msg->sample_rate) != sample_rate_ || msg->channels != 1) {
      RCLCPP_ERROR_THROTTLE(
        get_logger(), *get_clock(), 5000,
        "Ignoring incompatible /audio/raw format: %u Hz, %u channels",
        msg->sample_rate, static_cast<unsigned int>(msg->channels));
      return;
    }

    vad_pending_.reserve(vad_pending_.size() + msg->samples.size());
    for (auto s : msg->samples) {
      vad_pending_.push_back(static_cast<float>(s) / 32768.0f);
    }

    while (
      vad_pending_.size() >= VAD_FRAME_SAMPLES &&
      listening_.load() &&
      !latched_.load())
    {
      std::vector<float> frame(
        vad_pending_.begin(), vad_pending_.begin() + VAD_FRAME_SAMPLES);
      vad_pending_.erase(
        vad_pending_.begin(), vad_pending_.begin() + VAD_FRAME_SAMPLES);
      process_vad_frame(frame);
    }
  }

  void process_vad_frame(const std::vector<float> & frame) {
    const bool speech = whisper_vad_detect_speech_no_reset(
      vad_ctx_, frame.data(), static_cast<int>(frame.size()));

    if (!in_speech_) {
      if (!speech) {
        append_pre_roll(frame);
        return;
      }

      in_speech_ = true;
      speech_buffer_.insert(speech_buffer_.end(), pre_roll_.begin(), pre_roll_.end());
      speech_buffer_.insert(speech_buffer_.end(), frame.begin(), frame.end());
      voiced_samples_ = frame.size();
      silent_samples_ = 0;
      publish_state("speech");

      if (speech_buffer_.size() >= max_speech_samples_) {
        RCLCPP_WARN(
          get_logger(), "Maximum utterance duration %.2fs reached; transcribing",
          max_speech_s_);
        queue_transcription("maximum utterance duration");
      }
      return;
    }

    speech_buffer_.insert(speech_buffer_.end(), frame.begin(), frame.end());

    if (speech) {
      voiced_samples_ += frame.size();
      silent_samples_ = 0;
    } else {
      // Measure silence from the audio itself, not wall-clock callback timing.
      silent_samples_ += frame.size();
    }

    if (speech_buffer_.size() >= max_speech_samples_) {
      RCLCPP_WARN(
        get_logger(), "Maximum utterance duration %.2fs reached; transcribing",
        max_speech_s_);
      queue_transcription("maximum utterance duration");
      return;
    }

    if (speech) return;
    if (silent_samples_ < silence_timeout_samples_) return;

    queue_transcription("silence timeout");
  }

  void queue_transcription(const char * reason) {
    if (voiced_samples_ < min_voiced_samples_) {
      RCLCPP_DEBUG(
        get_logger(),
        "Discarding short speech candidate (%zu voiced samples < %zu minimum)",
        voiced_samples_, min_voiced_samples_);
      reset_capture();
      whisper_vad_reset_state(vad_ctx_);
      publish_state("listening");
      return;
    }

    latched_.store(true); // one LISTENING period -> one accepted utterance

    WorkItem item{session_.load(), std::move(speech_buffer_)};
    const double audio_s = static_cast<double>(item.audio.size()) / sample_rate_;

    reset_capture();

    // We use streaming VAD, so each accepted utterance must start with fresh LSTM state.
    whisper_vad_reset_state(vad_ctx_);
    publish_state("transcribing");

    {
      std::lock_guard<std::mutex> lock(work_mutex_);
      if (work_queue_.size() >= 2) {
        work_queue_.pop_front();
      }
      work_queue_.push_back(std::move(item));
    }
    work_cv_.notify_one();

    RCLCPP_INFO(
      get_logger(), "Utterance complete (%s, %.2fs audio); queued for transcription",
      reason, audio_s);
  }

  void append_pre_roll(const std::vector<float> & frame) {
    for (float s : frame) pre_roll_.push_back(s);
    while (pre_roll_.size() > pre_roll_max_samples_) pre_roll_.pop_front();
  }

  void reset_capture() {
    vad_pending_.clear();
    pre_roll_.clear();
    speech_buffer_.clear();
    voiced_samples_ = 0;
    silent_samples_ = 0;
    in_speech_ = false;
  }

  void purge_pending_work(const char * reason) {
    std::size_t dropped = 0;
    {
      std::lock_guard<std::mutex> lock(work_mutex_);
      dropped = work_queue_.size();
      work_queue_.clear();
    }

    if (dropped > 0) {
      RCLCPP_INFO(
        get_logger(), "Dropped %zu queued stale STT item(s): %s", dropped, reason);
    }
  }

  void worker_loop() {
    while (!stop_.load()) {
      WorkItem item;
      {
        std::unique_lock<std::mutex> lock(work_mutex_);
        work_cv_.wait(lock, [this] { return stop_.load() || !work_queue_.empty(); });
        if (stop_.load()) return;

        item = std::move(work_queue_.front());
        work_queue_.pop_front();
      }

      active_session_.store(item.session);

      const auto strategy = beam_size_ > 1 ?
        WHISPER_SAMPLING_BEAM_SEARCH : WHISPER_SAMPLING_GREEDY;
      auto p = whisper_full_default_params(strategy);
      p.n_threads = whisper_threads_;
      p.translate = false;
      p.no_context = true;
      p.no_timestamps = true;
      p.print_special = false;
      p.print_progress = false;
      p.print_realtime = false;
      p.print_timestamps = false;
      p.language = language_.c_str();

      if (strategy == WHISPER_SAMPLING_BEAM_SEARCH) {
        p.beam_search.beam_size = beam_size_;
      } else {
        p.greedy.best_of = 1;
      }

      p.abort_callback = &SpeechToTextNode::abort_callback;
      p.abort_callback_user_data = this;

      const auto start = std::chrono::steady_clock::now();
      const int rc = whisper_full(
        whisper_ctx_, p, item.audio.data(), static_cast<int>(item.audio.size()));
      const double infer_s = std::chrono::duration<double>(
        std::chrono::steady_clock::now() - start).count();

      if (rc != 0) {
        if (!stale(item.session)) {
          RCLCPP_ERROR(get_logger(), "whisper_full failed rc=%d", rc);
          allow_retry(item.session);
        }
        continue;
      }

      std::string text;
      const int n = whisper_full_n_segments(whisper_ctx_);
      for (int i = 0; i < n; ++i) {
        const char * raw = whisper_full_get_segment_text(whisper_ctx_, i);
        if (!raw) continue;

        auto part = trim(raw);
        if (part.empty() || part == ".") continue;
        if (!text.empty()) text += " ";
        text += part;
      }
      text = trim(text);

      if (stale(item.session) && !publish_stale_results_) {
        RCLCPP_INFO(
          get_logger(), "Dropping stale STT result from session %lu",
          static_cast<unsigned long>(item.session));
        continue;
      }

      if (!useful_text(text)) {
        RCLCPP_INFO(
          get_logger(),
          "No useful text; allowing another utterance in current LISTENING session");
        allow_retry(item.session);
        continue;
      }

      std_msgs::msg::String out;
      out.data = text;
      text_pub_->publish(out);
      publish_state("result");

      const double audio_s = static_cast<double>(item.audio.size()) / sample_rate_;
      RCLCPP_INFO(
        get_logger(),
        "Heard: %s (audio %.2fs, inference %.3fs, realtime factor %.2f)",
        text.c_str(), audio_s, infer_s, audio_s > 0.0 ? infer_s / audio_s : 0.0);
    }
  }

  static bool abort_callback(void * data) {
    auto * self = static_cast<SpeechToTextNode *>(data);
    if (self->stop_.load()) return true;
    if (self->publish_stale_results_) return false;

    return
      !self->listening_.load() ||
      self->session_.load() != self->active_session_.load();
  }

  bool stale(std::uint64_t s) const {
    return !listening_.load() || session_.load() != s;
  }

  void allow_retry(std::uint64_t s) {
    if (listening_.load() && session_.load() == s) {
      // Capture and VAD state were already reset when this utterance was queued.
      // Re-open the current LISTENING session only after Whisper has finished.
      latched_.store(false);
      publish_state("listening");
    }
  }

  void publish_state(const std::string & s) {
    if (!diagnostic_pub_) return;

    std_msgs::msg::String msg;
    msg.data = s;
    diagnostic_pub_->publish(msg);
  }

  std::string audio_topic_, effective_state_topic_, text_topic_, diagnostic_topic_;
  int sample_rate_{SAMPLE_RATE};
  double silence_timeout_{1.0}, max_speech_s_{30.0};
  int pre_roll_ms_{300}, min_speech_ms_{160};
  std::string model_path_, vad_model_path_, language_{"en"};
  bool use_gpu_{true}, flash_attn_{false}, publish_stale_results_{false};
  int gpu_device_{0}, whisper_threads_{4}, beam_size_{5}, vad_threads_{2};
  std::size_t pre_roll_max_samples_{0};
  std::size_t min_voiced_samples_{0};
  std::size_t silence_timeout_samples_{0};
  std::size_t max_speech_samples_{0};

  whisper_context * whisper_ctx_{nullptr};
  whisper_vad_context * vad_ctx_{nullptr};

  rclcpp::Publisher<std_msgs::msg::String>::SharedPtr text_pub_, diagnostic_pub_;
  rclcpp::Subscription<k9_interfaces_pkg::msg::AudioFrame>::SharedPtr audio_sub_;
  rclcpp::Subscription<std_msgs::msg::String>::SharedPtr state_sub_;

  std::string effective_state_;
  std::atomic<bool> listening_{false}, latched_{true}, stop_{false};
  std::atomic<std::uint64_t> session_{0}, active_session_{0};

  std::vector<float> vad_pending_, speech_buffer_;
  std::deque<float> pre_roll_;
  std::size_t voiced_samples_{0};
  std::size_t silent_samples_{0};
  bool in_speech_{false};

  std::mutex work_mutex_;
  std::condition_variable work_cv_;
  std::deque<WorkItem> work_queue_;
  std::thread worker_;
};

int main(int argc, char ** argv) {
  rclcpp::init(argc, argv);
  try {
    rclcpp::spin(std::make_shared<SpeechToTextNode>());
  } catch (const std::exception & e) {
    std::fprintf(stderr, "k9_stt fatal error: %s\n", e.what());
    rclcpp::shutdown();
    return 1;
  }
  rclcpp::shutdown();
  return 0;
}
