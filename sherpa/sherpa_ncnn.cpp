
#include <iostream>
#include <string>
#include "net.h"
#include "sherpa-ncnn/c-api/c-api.h"

void test_sherpa_ncnn() {

    std::string encode_param_file("/Users/yang/CLionProjects/test_ncnn2/sherpa/encoder_jit_trace-pnnx.ncnn.param");
    std::string encode_bin_file("/Users/yang/CLionProjects/test_ncnn2/sherpa/encoder_jit_trace-pnnx.ncnn.bin");
    std::string decode_param_file("/Users/yang/CLionProjects/test_ncnn2/sherpa/decoder_jit_trace-pnnx.ncnn.param");
    std::string decode_bin_file("/Users/yang/CLionProjects/test_ncnn2/sherpa/decoder_jit_trace-pnnx.ncnn.bin");
    std::string joiner_param_file("/Users/yang/CLionProjects/test_ncnn2/sherpa/joiner_jit_trace-pnnx.ncnn.param");
    std::string joiner_bin_file("/Users/yang/CLionProjects/test_ncnn2/sherpa/joiner_jit_trace-pnnx.ncnn.bin");
    std::string tokens_file("/Users/yang/CLionProjects/test_ncnn2/sherpa/tokens.txt");
    std::string wav_file = "/Users/yang/CLionProjects/test_ncnn2/data/audio/test_chinese_1.wav";

    SherpaNcnnRecognizerConfig config;
    memset(&config, 0, sizeof(config));

    config.model_config.tokens = tokens_file.c_str();
    config.model_config.encoder_param = encode_param_file.c_str();
    config.model_config.encoder_bin = encode_bin_file.c_str();
    config.model_config.decoder_param = decode_param_file.c_str();
    config.model_config.decoder_bin = decode_bin_file.c_str();
    config.model_config.joiner_param = joiner_param_file.c_str();
    config.model_config.joiner_bin = joiner_bin_file.c_str();
    config.model_config.num_threads = 4;
    config.model_config.use_vulkan_compute = 0;
    config.decoder_config.decoding_method = "greedy_search";
    config.decoder_config.num_active_paths = 4;
    config.enable_endpoint = 0;
    config.rule1_min_trailing_silence = 2.4;
    config.rule2_min_trailing_silence = 1.2;
    config.rule3_min_utterance_length = 300;

    config.feat_config.sampling_rate = 16000;
    config.feat_config.feature_dim = 80;
//    if (argc >= 12) {
//        config.hotwords_file = argv[11];
//    }

//    if (argc == 13) {
//        config.hotwords_score = atof(argv[12]);
//    }
//
    SherpaNcnnRecognizer *recognizer = CreateRecognizer(&config);

    const char *wav_filename = wav_file.c_str();
    FILE *fp = fopen(wav_filename, "rb");
    if (!fp) {
        fprintf(stderr, "Failed to open %s\n", wav_filename);
        return;
    }

    // Assume the wave header occupies 44 bytes.
    fseek(fp, 44, SEEK_SET);

    // simulate streaming
    int N = 3200;  // 0.2 s. Sample rate is fixed to 16 kHz

    int16_t buffer[N];
    float samples[N];
    SherpaNcnnStream *s = CreateStream(recognizer);

    SherpaNcnnDisplay *display = CreateDisplay(50);
    int32_t segment_id = -1;

    while (!feof(fp)) {
        size_t n = fread((void *)buffer, sizeof(int16_t), N, fp);
        if (n > 0) {
            for (size_t i = 0; i != n; ++i) {
                samples[i] = buffer[i] / 32768.;
            }
            AcceptWaveform(s, 16000, samples, n);
            while (IsReady(recognizer, s)) {
                Decode(recognizer, s);
            }

            SherpaNcnnResult *r = GetResult(recognizer, s);
            if (strlen(r->text)) {
                std::cout << r->text << std::endl;
//                SherpaNcnnPrint(display, segment_id, r->text);
            }
            DestroyResult(r);
        }
    }
    fclose(fp);

    // add some tail padding
//    float tail_paddings[4800] = {0};  // 0.3 seconds at 16 kHz sample rate
//    AcceptWaveform(s, 16000, tail_paddings, 4800);
//
//    InputFinished(s);
//
//    while (IsReady(recognizer, s)) {
//        Decode(recognizer, s);
//    }
//    SherpaNcnnResult *r = GetResult(recognizer, s);
//    if (strlen(r->text)) {
//        SherpaNcnnPrint(display, segment_id, r->text);
//    }
//
//    DestroyResult(r);

    DestroyDisplay(display);

    DestroyStream(s);
    DestroyRecognizer(recognizer);

    fprintf(stderr, "\n");

    return;

}