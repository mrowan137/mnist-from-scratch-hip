#include <algorithm>
#include <cctype>
#include <climits>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>

#include <hip/hip_runtime.h>

#define HIP_API_CHECK(call)                                                    \
  do {                                                                         \
    hipError_t err = call;                                                     \
    if (err != hipSuccess) {                                                   \
      std::cerr << "HIP error: " << hipGetErrorString(err) << " calling "      \
                << #call << " at " << __FILE__ << ": " << __LINE__ << '\n';    \
      std::exit(EXIT_FAILURE);                                                 \
    }                                                                          \
  } while (0)                                                                  \

#define HIP_KERNEL_CHECK(call)                                                 \
  call;                                                                        \
  do {                                                                         \
    hipError_t err = hipGetLastError();                                        \
    if (err != hipSuccess) {                                                   \
      std::cerr << "HIP error: " << hipGetErrorString(err) << " calling "      \
                << #call << " at " << __FILE__ << ": " << __LINE__ << '\n';    \
      std::exit(EXIT_FAILURE);                                                 \
    }                                                                          \
  } while (0)

#if defined(__GFX8__) || defined(__GFX9__)
static constexpr int WAVEFRONT_SIZE = 64;
#else  // gfx10xx, gfx11xx, gfx12xx, ...
static constexpr int WAVEFRONT_SIZE = 32;
#endif

static constexpr int THREADS_PER_BLOCK = 64;
static constexpr int BLOCKS_PER_GRID = 128;

static constexpr int MT = 10;
static constexpr int NT = 100;
static constexpr int KT = 25;
static constexpr int MTi = 16;
static constexpr int KTi = 32;
static constexpr int NTi = 150;

struct Config {
  /*
    Parse and check command line arguments.
  */
  std::string dataset_path;
  int iters = 100;
  float lr = 0.01f;

  Config (int argc, char** argv) {
    if (argc < 2 || argc > 4) {
      std::cerr << "Usage: ./mnist /path/to/dataset [iters] [learning rate]\n";
      std::exit(EXIT_FAILURE);
    }

    dataset_path = argv[1];
    if (!std::filesystem::exists(dataset_path)) {
      std::cerr << "Dataset path \'" << dataset_path << "\' does not exist!\n";
      std::exit(EXIT_FAILURE);
    }

    if (argc > 2
        && (sscanf(argv[2], "%d", &iters) != 1
            || !(1 <= iters && iters <= INT_MAX))) {
      std::cerr << "Iterations should be a positive integer.\n";
      std::exit(EXIT_FAILURE);
    }

    if (argc > 3
        && (sscanf(argv[3], "%f", &lr) != 1
            || lr < 0.0f)) {
      std::cerr << "Learning rate should be a non-negative float.\n";
      std::exit(EXIT_FAILURE);
    }
  }
};

struct MNIST {
  /*
    Wrapper to hold MNIST images and label data.
  */
  Config config;
  static constexpr int k_num_categories = 10;  // 10 handwritten digits
  int num_train_images, num_test_images;
  int rows, cols, pixels_per_image;
  std::vector<uint8_t> x_train, y_train, yhat_train;
  std::vector<uint8_t> x_test, y_test, yhat_test;
  std::vector<float> x_train_flt, y_train_flt;
  std::vector<float> x_train_T_flt;
  std::vector<float> x_test_flt, y_test_flt;

  enum class DatasetType {
    TEST,
    TRAIN
  };

  int reverse_int (const int n) {
    uint8_t bytes[4] = {};
    bytes[0] = n & 0xFF;
    bytes[1] = (n >> 8) & 0xFF;
    bytes[2] = (n >> 16) & 0xFF;
    bytes[3] = (n >> 24) & 0xFF;
    return ((static_cast<int>(bytes[0]) << 24)
            + (static_cast<int>(bytes[1]) << 16)
            + (static_cast<int>(bytes[2]) << 8)
            + static_cast<int>(bytes[3]));
  }

  void read_file (std::ifstream& file, char* dst, const std::streamsize n,
                  const std::string& path, const std::string& what) {
    file.read(dst, n);
    if (!file) {
      std::cerr << "Unable to read " << what << " from path: " << path << "\n";
      std::exit(EXIT_FAILURE);
    }
  }

  std::vector<uint8_t> read_mnist_labels (const std::string& mnist_labels_path) {
    int magic = 0, num_labels = 0;
    std::vector<uint8_t> labels;
    std::ifstream file(mnist_labels_path, std::ios::binary);

    if (file.is_open()) {
      read_file(file, reinterpret_cast<char*>(&magic), sizeof(magic),
                mnist_labels_path, "labels magic");
      magic = reverse_int(magic);
      if (magic != 2049) {
        std::cerr << "Invalid MNIST labels dataset: " << mnist_labels_path << "\n";
        std::cerr << "Expected magic == 2049, read magic == " << magic << "\n";
        std::exit(EXIT_FAILURE);
      }

      read_file(file, reinterpret_cast<char*>(&num_labels), sizeof(num_labels),
                mnist_labels_path, "number of labels");
      num_labels = reverse_int(num_labels);

      labels.resize(num_labels);
      read_file(file, reinterpret_cast<char*>(labels.data()), num_labels,
                mnist_labels_path, "labels data");
    } else {
      std::cerr << "Unable to open MNIST labels file: " << mnist_labels_path << "\n";
      std::exit(EXIT_FAILURE);
    }
    return labels;
  }

  std::vector<uint8_t> read_mnist_images (const std::string& mnist_images_path) {
    int magic = 0, num_images = 0;
    std::vector<uint8_t> images;
    std::ifstream file(mnist_images_path, std::ios::binary);

    if (file.is_open()) {
      read_file(file, reinterpret_cast<char*>(&magic), sizeof(magic),
                mnist_images_path, "images magic");
      magic = reverse_int(magic);
      if (magic != 2051) {
        std::cerr << "Invalid MNIST images dataset: " << mnist_images_path << "\n";
        std::cerr << "Expected magic == 2051, read magic == " << magic << "\n";
        std::exit(EXIT_FAILURE);
      }

      read_file(file, reinterpret_cast<char*>(&num_images), sizeof(num_images),
                mnist_images_path, "number of images");
      num_images = reverse_int(num_images);
      read_file(file, reinterpret_cast<char*>(&rows), sizeof(rows),
                mnist_images_path, "number of rows");
      rows = reverse_int(rows);
      read_file(file, reinterpret_cast<char*>(&cols), sizeof(cols),
                mnist_images_path, "number of cols");
      cols = reverse_int(cols);

      uint32_t count = num_images * rows * cols;
      images.resize(count);
      read_file(file, reinterpret_cast<char*>(images.data()), count,
                mnist_images_path, "images data");
    } else {
      std::cerr << "Unable to open MNIST images file: " << mnist_images_path << "\n";
      std::exit(EXIT_FAILURE);
    }
    return images;
  }

  void load_data () {
    x_test = read_mnist_images(config.dataset_path + "/t10k-images-idx3-ubyte");
    x_train = read_mnist_images(config.dataset_path + "/train-images-idx3-ubyte");
    y_test = read_mnist_labels(config.dataset_path + "/t10k-labels-idx1-ubyte");
    y_train = read_mnist_labels(config.dataset_path + "/train-labels-idx1-ubyte");

    pixels_per_image = rows * cols;
    num_train_images = x_train.size() / pixels_per_image;
    num_test_images = x_test.size() / pixels_per_image;

    x_train_flt.reserve(x_train.size() + x_train.size() / pixels_per_image);  // extra space to absorb bias in W
    for (int i = 0; i < x_train.size(); i += pixels_per_image) {
      for (int j = 0; j < pixels_per_image; ++j) {
        x_train_flt.push_back(static_cast<float>(x_train[i + j]) / 255.0f);  // scale pixel value in preparation for training
      }
      x_train_flt.push_back(1.0f);
    }

    x_train_T_flt.reserve(x_train.size() + x_train.size() / pixels_per_image);  // for coalesced memory access
    for (int j = 0; j < pixels_per_image; ++j) {
      for (int i = 0; i < x_train.size(); i += pixels_per_image) {
        x_train_T_flt.push_back(static_cast<float>(x_train[i + j]) / 255.0f);
      }
    }
    for (int j = 0; j < num_train_images; ++j) x_train_T_flt.push_back(1.0f);

    y_train_flt.resize(y_train.size(), 0.0f);
    for (int i = 0; i < y_train.size(); ++i) {
      y_train_flt[i] = static_cast<float>(y_train[i]);
    }

    x_test_flt.reserve(x_test.size() + x_test.size() / pixels_per_image);
    for (int i = 0; i < x_test.size(); i += pixels_per_image) {
      for (int j = 0; j < pixels_per_image; ++j) {
        x_test_flt.push_back(static_cast<float>(x_test[i + j]) / 255.0f);
      }
      x_test_flt.push_back(1.0f);
    }

    y_test_flt.resize(y_test.size(), 0.0f);
    for (int i = 0; i < y_test.size(); ++i) {
      y_test_flt[i] = static_cast<float>(y_test[i]);
    }
    std::cout << "\nMNIST dataset\n";
    std::cout << "  Image size: " << rows << " x " << cols << "\n";
    std::cout << "  Training images: " << num_train_images << "\n";
    std::cout << "  Test images: " << num_test_images << "\n";
  }

  void print_image (int image, const DatasetType dataset_type = DatasetType::TRAIN) {
    const char* ramp = "$@B%8&WM#*oahkbdpqwmZO0QLCJUYXzcvunxrjft/\\|()1{}[]?-_+~<>i!lI;:,\"^`'. ";
    const std::vector<uint8_t>& x = (dataset_type == DatasetType::TRAIN ? x_train : x_test);  // ^70 values, dark --> light
    const std::vector<uint8_t>& y = (dataset_type == DatasetType::TRAIN ? y_train : y_test);
    const std::vector<uint8_t>& yhat = (dataset_type == DatasetType::TRAIN ? yhat_train : yhat_test);
    const int num_images = (dataset_type == DatasetType::TRAIN ? num_train_images : num_test_images);
    size_t width = 2 * rows + 2;

    std::string cmd;
    while (true) {
      const std::string annotation = (dataset_type == DatasetType::TRAIN ? "training" : "test");
      const std::string dataset_msg = ("DATASET: " + annotation + "\n");
      const std::string label_msg = ("TRUTH: " + std::to_string(y[image]) + "\n");
      const std::string pred_msg = !yhat.empty() ? ("PREDICTION: " + std::to_string(yhat[image])) : "";
      const std::string id_msg = "ID: " + std::to_string(image + 1) + "/" + std::to_string(num_images) + "\n";
      std::cout << "\n" + std::string(std::max(static_cast<int>(width - dataset_msg.size()) / 2, 1), ' ') + dataset_msg;
      std::cout << std::string(std::max(static_cast<int>(width - label_msg.size()) / 2, 1), ' ') + label_msg;
      std::cout << std::string(std::max(static_cast<int>(width - id_msg.size()) / 2, 1), ' ') + id_msg;

      if (!pred_msg.empty()) {
        bool correct = (yhat[image] == y[image]);
        std::cout << std::string(std::max(static_cast<int>(width - pred_msg.size()) / 2, 1), ' ')
                     + (correct ? "\033[48;2;40;80;40m" + pred_msg + "\033[0m\n"
                                : "\033[48;2;90;35;35m" + pred_msg + "\033[0m\n");
      }

      for (int i = 0; i <= width / 2; ++i) std::cout << "* ";
      std::cout << "\n";
      for (int i = 0; i < rows; ++i) {
        std::cout << "* ";
        for (int j = 0; j < cols; ++j) {
          int idx = i * cols + j;
          int v = 69 * x[image * (rows * cols) + idx] / 255.0f;
          std::cout << ramp[69 - v] << " ";
        }
        std::cout << "*\n";
      }
      for (int i = 0; i <= width / 2; ++i) std::cout << "* ";
      std::cout << "\n";

      std::cout << "[n]ext    [p]rev    [c]ontinue\n";
      std::cin >> cmd;

      if (cmd == "c") break;

      auto is_number = [](const std::string& num) {  // check if non-negative number
        return std::all_of(num.begin(), num.end(),
                           [&](char c){
                             return std::isdigit(static_cast<unsigned char>(c));
                           });
      };
      if (is_number(cmd)) {
        image = std::atoi(cmd.c_str()) - 1;
        image = std::min(num_images - 1, image);
        image = std::max(0, image);
        continue;
      }
      // negative input interpreted as 'p'
      image = (image + (cmd == "n" ? 1 : -1) + num_images) % num_images;
    }
  }

  MNIST (const Config& config_)
    : config(config_) {
    load_data();
  }
};


__device__ float random_flt (uint32_t seed) {
  /*
    Caller responsible for maintaining seed.
    Returns pseudo random number on [0,1).
    Reference: github.com/joelkp/ranoise/blob/main/splitmix32.c
  */
  seed = (seed ^ (seed >> 16)) * 0x85ebca6b;
  seed = (seed ^ (seed >> 13)) * 0xc2b2ae35;
  seed = (seed ^ (seed >> 16));
  return (static_cast<float>(seed >> 8)) / (static_cast<float>(1u << 24));  // match significant digits of IEEE float
}

__launch_bounds__(THREADS_PER_BLOCK, 1)
__global__ void initialize_weights (float* d_weights, const int M, const int K) {
  int idx = blockDim.x * blockIdx.x + threadIdx.x;
  int stride = gridDim.x * blockDim.x;
  for (int i = idx; i < M * K; i += stride) {
    d_weights[i] = 0.01f * (random_flt(static_cast<uint32_t>(i) + 1234567) - 0.5f);
  }
}

template <int MT, int NT, int KT>
__launch_bounds__(MT * NT, 1)
__global__ void sgemm_sub_and_scale (const float* __restrict__ A, const float* __restrict__ B,
                                     const float* __restrict__ C, float* __restrict__ D,
                                     const int M, const int N, const int K,
                                     const float alpha) {
  /*
    Computes D = alpha * (A @ Bt - C).
    A, B laid out as M x K, N x K.
    C stores class labels as 1 x N; one-hot columns constructed on the fly.

    (1) Yhat = W @ Xt,                             [ M x N = (M x K) @ (K x N) ]
        and Yhat --> 2 * (Yhat - Y) = 2 * (W @ Xt - Y), in preparation for (2).
  */
  const int ti = threadIdx.x, tj = threadIdx.y;
  const int bi0 = blockIdx.x * MT, bj0 = blockIdx.y * NT;
  const int gidx = (bi0 + ti) * N + (bj0 + tj);  // global index in D
  const int block_stride = blockDim.x * blockDim.y;
  const int bidx = tj * blockDim.x + ti;

  __shared__ float As[2][MT * KT];
  __shared__ float Bs[2][NT * KT];
  int compute = 0, load = 1;

  int bik0 = bi0 * K + 0;
  int bjk0 = bj0 * K + 0;
  for (int idx = bidx; idx < MT * KT; idx += block_stride) {
    As[compute][idx] = (0 + (idx % KT) < K && (bi0 + (idx / KT)) < M)
                         ? A[bik0 + (idx / KT) * K + idx % KT] : 0.0f;
  }
  for (int idx = bidx; idx < KT * NT; idx += block_stride) {
    Bs[compute][idx] = (0 + (idx % KT) < K && (bj0 + (idx / KT)) < N)
                         ? B[bjk0 + (idx / KT) * K + idx % KT] : 0.0f;
  }
  __syncthreads();

  float acc = 0.0f;
  for (int bk = KT; bk < K; bk += KT) {
    int bik0 = bi0 * K + bk;
    int bjk0 = bj0 * K + bk;

    for (int idx = bidx; idx < MT * KT; idx += block_stride) {
      As[load][idx] = (bk + (idx % KT) < K && (bi0 + (idx / KT)) < M)
                        ? A[bik0 + (idx / KT) * K + idx % KT] : 0.0f;
    }
    for (int idx = bidx; idx < KT * NT; idx += block_stride) {
      Bs[load][idx] = (bk + (idx % KT) < K && (bj0 + (idx / KT)) < N)
                        ? B[bjk0 + (idx / KT) * K + idx % KT] : 0.0f;
    }

#pragma unroll
    for (int tk = 0; tk < KT; ++tk) {
      acc += As[compute][ti * KT + tk] * Bs[compute][tj * KT + tk];
    }

    __syncthreads();
    compute ^= 1;
    load ^= 1;
  }

#pragma unroll
  for (int tk = 0; tk < KT; ++tk) {
    acc += As[compute][ti * KT + tk] * Bs[compute][tj * KT + tk];
  }

  if (bi0 + ti < M && bj0 + tj < N) D[gidx] = alpha * (acc - (gidx / N == C[gidx % N] ? 1.0f : 0.0f));
}

template <int MT, int NT, int KT>
__launch_bounds__(MTi * KTi, 1)
__global__ void sgemm (const float* __restrict__ A, const float* __restrict__ B,
                       float* __restrict__ C,
                       const int M, const int N, const int K) {
  /*
    Computes C = A @ Bt.
    A, B laid out as M x K, N x K.

    (2) Grad = 2 * (Yhat - Ytrue) @ X              [ M x K = (M x N) @ (N x K) ]

                                              N
                                      NT
                                   KT . . . . x x x x
                                      . . . . x x x x      * * *
                                      . . . . x x x x      * B *
                                 K    x x x x x x x x      * * *
                                      x x x x x x x x
                                      x x x x x x x x
                         K
                   KT                 NT
    * * *       MT . . . x x x     MT . . . . x x x x      * * *
    * A *          . . . x x x        . . . . x x x x      * C *
    * * *    M     x x x x x x        x x x x x x x x  M   * * *
                   x x x x x x        x x x x x x x x

                                              N

  */

  const int ti = threadIdx.x, tj = threadIdx.y;            // coords in MT x NT tile
  const int bi0 = blockIdx.x * MT, bj0 = blockIdx.y * NT;  // corner of this block
  const int gidx = (bi0 + ti) * N + (bj0 + tj);            // global index in C
  const int bidx = tj * blockDim.x + ti;
  const int block_stride = blockDim.x * blockDim.y;

  // Bs laid out as NT x KT so threads access elements of Bs contiguously
  __shared__ float As[2][MT * KT];  // [2] for ping-pong buffers
  __shared__ float Bs[2][NT * KT];
  int compute = 0, load = 1;

  // initial load of A_k( = 0), B_k( = 0) for buffer ping-pong
  int bik0 = bi0 * K + 0;  // linear idx to corner of current MT x KT chunk in A
  int bjk0 = bj0 * K + 0;  // linear idx to corner of current KT x NT chunk in B
  for (int idx = bidx; idx < MT * KT; idx += block_stride) {
    As[compute][idx] = (0 + (idx % KT) < K && (bi0 + (idx / KT)) < M)
                         ? A[bik0 + (idx / KT) * K + idx % KT] : 0.0f;
  }
  for (int idx = bidx; idx < KT * NT; idx += block_stride) {
    Bs[compute][idx] = (0 + (idx % KT) < K && (bj0 + (idx / KT)) < N)
                         ? B[bjk0 + (idx / KT) * K + idx % KT] : 0.0f;
  }
  __syncthreads();

  float acc = 0.0f;
  for (int bk = KT; bk < K; bk += KT) {
    int bik0 = bi0 * K + bk;
    int bjk0 = bj0 * K + bk;

    // load A_{k + 1}, B_{k + 1}
    for (int idx = bidx; idx < MT * KT; idx += block_stride) {
      As[load][idx] = (bk + (idx % KT) < K && (bi0 + (idx / KT)) < M)
                        ? A[bik0 + (idx / KT) * K + idx % KT] : 0.0f;
    }
    for (int idx = bidx; idx < KT * NT; idx += block_stride) {
      Bs[load][idx] = (bk + (idx % KT) < K && (bj0 + (idx / KT)) < N)
                        ? B[bjk0 + (idx / KT) * K + idx % KT] : 0.0f;
    }

    // compute A_k @ B_k
#pragma unroll
    for (int tk = 0; tk < KT; ++tk) {
      acc += As[compute][ti * KT + tk] * Bs[compute][tj * KT + tk];
    }

    // swap once load is ready
    __syncthreads();
    compute ^= 1;
    load ^= 1;
  }

  // final A_k, B_k contrib
#pragma unroll
  for (int tk = 0; tk < KT; ++tk) {
    acc += As[compute][ti * KT + tk] * Bs[compute][tj * KT + tk];
  }

  if (bi0 + ti < M && bj0 + tj < N) C[gidx] = acc;
}

__launch_bounds__(THREADS_PER_BLOCK, 1)
__global__ void update_weights (float* d_weights, const float* d_grads,
                                const float* d_yhat_train_scores, float* d_loss,
                                const int M, const int N, const int K,
                                const int it, const float lr) {
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  const int grid_stride = gridDim.x * blockDim.x;

  // (3) W -= learning_rate * Grad / len(X)            [ M x K =  M x K ]
  for (int i = idx; i < M * K; i += grid_stride) d_weights[i] -= lr * d_grads[i] / N;

  // (4) Loss = (Yhat - Ytrue)^2                       [ element-wise on M x N ]
  // (5) <Loss> = sum(Loss) / len(X)                   [ scalar ]

  // Yhat --> 2 * (Yhat - Y) is still in effect here, so Loss --> (Yhat / 2)^2
  float acc = 0.0f;
  for (int i = idx; i < M * N; i += grid_stride) {
    acc += (0.5f * d_yhat_train_scores[i]) * (0.5f * d_yhat_train_scores[i]);
  }

#pragma unroll  // warp reduce
  for (uint16_t w = (WAVEFRONT_SIZE >> 1); w > 0; w >>= 1) acc += __shfl_down(acc, w);

  if ((threadIdx.x % WAVEFRONT_SIZE) == 0) {
    atomicAdd(&d_loss[it], acc / N);
  }
}

__global__ void print_loss (const float* d_loss, const int it) {
  if (blockIdx.x == 0 && threadIdx.x == 0) {
    printf("  iteration: %d; loss: %.6f\n", it + 1, d_loss[it]);
  }
}

__launch_bounds__(THREADS_PER_BLOCK, 1)
__global__ void calc_num_correct (const float* d_yhat_train_scores, const float* d_y_train,
                                  uint8_t* d_yhat_train, int* d_num_correct,
                                  const int M, const int N) {
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  const int grid_stride = gridDim.x * blockDim.x;

  // Yhat --> 2 * (Yhat - Y) is still in effect here, so argmax(Yhat) --> argmax(Yhat / 2 + Y)
  for (int j = idx; j < N; j += grid_stride) {
    int argmax = -1;
    float max_so_far = -INFINITY;
    for (int i = 0; i < M; ++i) {
      float m = 0.5f * d_yhat_train_scores[i * N + j] + (i == static_cast<int>(d_y_train[j]) ? 1.0f : 0.0f);
      if (m > max_so_far) {
        argmax = i;
        max_so_far = m;
      }
    }
    if (argmax == d_y_train[j]) {
      atomicAdd(d_num_correct, 1);
    }

    d_yhat_train[j] = static_cast<uint8_t>(argmax);
  }
}

struct Model {
  /*
    Model lives on the device.
    Owns model weights, gradients, and device copies of MNIST data.
    Runs the training loop and test.

    (M, N, K) := (10, num_images, 28 * 28 + 1)

    Weights:                          Xt (each col ~ image):      Y :

         |-- 28 * 28 + 1 --|                |-- num_images --|        |-- num_images --|

    ---   w00 w01 ... w0K b0          ---    x00 x01 ... x0N      ---  y00 y01 ... y0N
     |    w10 w11 ... w1K b1           |     x10 x11 ... x1N       |   y10 y11 ... y1N
     |    .   .       .   .            |     .   .       .         |   .   .       .
    10    .      .    .   .     28 * 28 + 1  .      .    .        10   .      .    .
     |    .         . .   .            |     .         . .         |   .         . .
    ---   wM0 wM1 ... wMK bM           |     xK0 xK1 ... xKN      ---  yM0 yM1 ... yMN
                                      ---    1   1   ... 1
  */

  // training hyperparameters
  const int iters = 100;
  const float lr = 0.01f;

  float training_elapsed_ms = 0.0f;

  static constexpr int k_threads_per_block = THREADS_PER_BLOCK;
  static constexpr int k_num_blocks = BLOCKS_PER_GRID;

  float* d_weights;                // M       x K
  float* d_grads;                  // M       x K
  float* d_x_train;                // N_train x K
  float* d_x_train_T;              // K x N_train
  float* d_x_test;                 // N_test  x K
  float* d_y_train;                // 1       x N_train
  float* d_y_test;                 // 1       x N_test
  float* d_yhat_train_scores;      // M       x N_train
  float* d_yhat_test_scores;       // M       x N_test
  float* d_loss;                   // 1       x iters
  float* d_final_loss;             // scalar
  int* d_num_correct;              // scalar (tally correct predictions)
  uint8_t* d_yhat_train;           // 1       x N_train (model predictions)
  uint8_t* d_yhat_test;            // 1       x N_test

  MNIST* mnist;

  void initialize_device_buffers () {
    /*
      Allocate all needed device buffers and zero them.
      Copy host data to device buffers.
    */
    const int x_train_numels = mnist->num_train_images * (mnist->pixels_per_image + 1);  // +1: bias
    const int x_test_numels = mnist->num_test_images * (mnist->pixels_per_image + 1);
    const int y_train_numels = mnist->num_train_images;
    const int y_test_numels = mnist->num_test_images;
    const int yhat_train_scores_numels = mnist->k_num_categories * mnist->num_train_images;
    const int yhat_test_scores_numels = mnist->k_num_categories * mnist->num_test_images;
    const int weights_numels = mnist->k_num_categories * (mnist->pixels_per_image + 1);
    const int grads_numels = mnist->k_num_categories * (mnist->pixels_per_image + 1);
    const int loss_numels = iters;
    HIP_API_CHECK(hipMalloc(&d_x_train, x_train_numels * sizeof(float))); HIP_API_CHECK(hipMemset(d_x_train, 0, x_train_numels * sizeof(float)));
    HIP_API_CHECK(hipMalloc(&d_x_train_T, x_train_numels * sizeof(float))); HIP_API_CHECK(hipMemset(d_x_train_T, 0, x_train_numels * sizeof(float)));
    HIP_API_CHECK(hipMalloc(&d_x_test, x_test_numels * sizeof(float))); HIP_API_CHECK(hipMemset(d_x_test, 0, x_test_numels * sizeof(float)));
    HIP_API_CHECK(hipMalloc(&d_y_train, y_train_numels * sizeof(float))); HIP_API_CHECK(hipMemset(d_y_train, 0, y_train_numels * sizeof(float)));
    HIP_API_CHECK(hipMalloc(&d_y_test, y_test_numels * sizeof(float))); HIP_API_CHECK(hipMemset(d_y_test, 0, y_test_numels * sizeof(float)));
    HIP_API_CHECK(hipMalloc(&d_yhat_train_scores, yhat_train_scores_numels * sizeof(float))); HIP_API_CHECK(hipMemset(d_yhat_train_scores, 0, yhat_train_scores_numels * sizeof(float)));
    HIP_API_CHECK(hipMalloc(&d_yhat_test_scores, yhat_test_scores_numels * sizeof(float))); HIP_API_CHECK(hipMemset(d_yhat_test_scores, 0, yhat_test_scores_numels * sizeof(float)));
    HIP_API_CHECK(hipMalloc(&d_weights, weights_numels * sizeof(float))); HIP_API_CHECK(hipMemset(d_weights, 0, weights_numels * sizeof(float)));
    HIP_API_CHECK(hipMalloc(&d_grads, grads_numels * sizeof(float))); HIP_API_CHECK(hipMemset(d_grads, 0, grads_numels * sizeof(float)));
    HIP_API_CHECK(hipMalloc(&d_loss, loss_numels * sizeof(float))); HIP_API_CHECK(hipMemset(d_loss, 0, loss_numels * sizeof(float)));
    HIP_API_CHECK(hipMalloc(&d_final_loss, sizeof(float))); HIP_API_CHECK(hipMemset(d_final_loss, 0, sizeof(float)));
    HIP_API_CHECK(hipMalloc(&d_num_correct, sizeof(int))); HIP_API_CHECK(hipMemset(d_num_correct, 0, sizeof(int)));
    HIP_API_CHECK(hipMalloc(&d_yhat_train, mnist->num_train_images * sizeof(uint8_t))); HIP_API_CHECK(hipMemset(d_yhat_train, 0, mnist->num_train_images * sizeof(uint8_t)));
    HIP_API_CHECK(hipMalloc(&d_yhat_test, mnist->num_test_images * sizeof(uint8_t))); HIP_API_CHECK(hipMemset(d_yhat_test, 0, mnist->num_test_images * sizeof(uint8_t)));

    HIP_API_CHECK(hipMemcpy(d_x_train, mnist->x_train_flt.data(), mnist->x_train_flt.size() * sizeof(float), hipMemcpyHostToDevice));
    HIP_API_CHECK(hipMemcpy(d_x_train_T, mnist->x_train_T_flt.data(), mnist->x_train_T_flt.size() * sizeof(float), hipMemcpyHostToDevice));
    HIP_API_CHECK(hipMemcpy(d_x_test, mnist->x_test_flt.data(), mnist->x_test_flt.size() * sizeof(float), hipMemcpyHostToDevice));
    HIP_API_CHECK(hipMemcpy(d_y_train, mnist->y_train_flt.data(), mnist->y_train_flt.size() * sizeof(float), hipMemcpyHostToDevice));
    HIP_API_CHECK(hipMemcpy(d_y_test, mnist->y_test_flt.data(), mnist->y_test_flt.size() * sizeof(float), hipMemcpyHostToDevice));

    HIP_API_CHECK(hipDeviceSynchronize());
  }

  Model (MNIST* mnist_, const int iters_ = 100, const float lr_ = 1e-2)
    : iters(iters_), lr(lr_), mnist(mnist_) {
    initialize_device_buffers();
  }

  ~Model () {
    HIP_API_CHECK(hipFree(d_x_train));
    HIP_API_CHECK(hipFree(d_x_train_T));
    HIP_API_CHECK(hipFree(d_x_test));
    HIP_API_CHECK(hipFree(d_y_train));
    HIP_API_CHECK(hipFree(d_y_test));
    HIP_API_CHECK(hipFree(d_yhat_train_scores));
    HIP_API_CHECK(hipFree(d_yhat_test_scores));
    HIP_API_CHECK(hipFree(d_weights));
    HIP_API_CHECK(hipFree(d_grads));
    HIP_API_CHECK(hipFree(d_loss));
    HIP_API_CHECK(hipFree(d_final_loss));
    HIP_API_CHECK(hipFree(d_num_correct));
    HIP_API_CHECK(hipFree(d_yhat_train));
    HIP_API_CHECK(hipFree(d_yhat_test));
  }

  template <bool VerboseLoss = false>
  void train (const std::string& csv_path = "") {
    /*
      Complete forward, backward pass loop for iterations to train the model.
      Operates on the weights using training dataset.

      (1) Yhat = W @ Xt                            [ M x N = (M x K) @ (K x N) ]
      (2) Grad = 2 * (Yhat - Ytrue) @ X            [ M x K = (M x N) @ (N x K) ]
      (3) W -= learning_rate * Grad / len(X)       [ M x K =  M x K ]
      (4) Loss = (Yhat - Ytrue)^2                  [ element-wise on M x N ]
      (5) <Loss> = sum(Loss) / len(X)              [ scalar ]
    */
    std::cout << "\nTraining model...\n";
    std::cout << "  Iterations: " << iters << "\n";
    std::cout << "  Learning rate: " << lr << "\n";

    std::ofstream loss_csv;
    if (!csv_path.empty()) {
      loss_csv.open(csv_path);
      if (!loss_csv.is_open()) {
        std::cerr << "Unable to open CSV output file: " << csv_path << "\n";
        std::exit(EXIT_FAILURE);
      }
      loss_csv << std::fixed << std::setprecision(6);
      loss_csv << "iteration,loss\n";
    }

    hipEvent_t start, stop;
    HIP_API_CHECK(hipEventCreate(&start));
    HIP_API_CHECK(hipEventCreate(&stop));

    HIP_API_CHECK(hipEventRecord(start));
    HIP_KERNEL_CHECK(hipLaunchKernelGGL(
      initialize_weights,
      dim3(k_num_blocks, 1, 1), dim3(k_threads_per_block, 1, 1), 0, 0,
      d_weights, mnist->k_num_categories, mnist->pixels_per_image + 1));
    for (int it = 0; it < iters; ++it) {
      HIP_KERNEL_CHECK(hipLaunchKernelGGL(  // forward pass: (1) Yhat = W @ Xt
        (sgemm_sub_and_scale<MT, NT, KT>),
        dim3((mnist->k_num_categories + MT - 1) / MT, (mnist->num_train_images + NT - 1) / NT, 1), dim3(MT, NT, 1), 0, 0,
        d_weights, d_x_train, d_y_train, d_yhat_train_scores,
        mnist->k_num_categories, mnist->num_train_images, mnist->pixels_per_image + 1,
        2.0f));
      HIP_KERNEL_CHECK(hipLaunchKernelGGL(  // backward pass (gradients): (2) Grad = 2 * (Yhat - Ytrue) @ X
        (sgemm<MTi, KTi, NTi>),
        dim3((mnist->k_num_categories + MTi - 1) / MTi, (mnist->pixels_per_image + 1 + KTi - 1) / KTi, 1), dim3(MTi, KTi, 1), 0, 0,
        d_yhat_train_scores, d_x_train_T, d_grads,
        mnist->k_num_categories, mnist->pixels_per_image + 1, mnist->num_train_images));
      HIP_KERNEL_CHECK(hipLaunchKernelGGL(  // backward pass (update weights): (3) W -= learning_rate * Grad / len(X)
        update_weights,                     // compute loss: (4) Loss = (Yhat - Ytrue)^2, (5) <Loss> = sum(Loss) / len(X)
        dim3(k_num_blocks, 1, 1), dim3(k_threads_per_block, 1, 1), 0, 0,
        d_weights, d_grads, d_yhat_train_scores, d_loss,
        mnist->k_num_categories, mnist->num_train_images, mnist->pixels_per_image + 1,
        it, lr));
      if constexpr (VerboseLoss) {
        HIP_KERNEL_CHECK(hipLaunchKernelGGL(print_loss,
                                            dim3(1, 1, 1), dim3(1, 1, 1), 0, 0,
                                            d_loss, it));
      }
    }

    HIP_API_CHECK(hipEventRecord(stop));
    HIP_API_CHECK(hipEventSynchronize(stop));

    // run these one more time to get forward pass with current weights and final loss
    HIP_KERNEL_CHECK(hipLaunchKernelGGL(
      (sgemm_sub_and_scale<MT, NT, KT>),
      dim3((mnist->k_num_categories + MT - 1) / MT,(mnist->num_train_images + NT - 1) / NT, 1), dim3(MT, NT, 1), 0, 0,
      d_weights, d_x_train, d_y_train, d_yhat_train_scores,
      mnist->k_num_categories, mnist->num_train_images, mnist->pixels_per_image + 1,
      2.0f));
    HIP_API_CHECK(hipMemset(d_num_correct, 0, sizeof(int)));
    HIP_KERNEL_CHECK(hipLaunchKernelGGL(calc_num_correct,
                                        dim3(k_num_blocks, 1, 1), dim3(k_threads_per_block, 1, 1), 0, 0,
                                        d_yhat_train_scores, d_y_train, d_yhat_train, d_num_correct,
                                        mnist->k_num_categories, mnist->num_train_images));
    HIP_API_CHECK(hipMemset(d_final_loss, 0, sizeof(float)));
    HIP_KERNEL_CHECK(hipLaunchKernelGGL(
      update_weights,
      dim3(k_num_blocks, 1, 1), dim3(k_threads_per_block, 1, 1), 0, 0,
      d_weights, d_grads, d_yhat_train_scores, d_final_loss,
      mnist->k_num_categories, mnist->num_train_images, mnist->pixels_per_image + 1,
      0, 0));  // lr = 0 to skip updating weights
    HIP_API_CHECK(hipDeviceSynchronize());

    training_elapsed_ms = 0.0f;
    HIP_API_CHECK(hipEventElapsedTime(&training_elapsed_ms, start, stop));

    HIP_API_CHECK(hipEventDestroy(start));
    HIP_API_CHECK(hipEventDestroy(stop));

    // copy out predictions to be ready for prediction labels in visualization
    mnist->yhat_train.resize(mnist->num_train_images, 0);
    HIP_API_CHECK(hipMemcpy(mnist->yhat_train.data(), d_yhat_train, mnist->num_train_images * sizeof(uint8_t), hipMemcpyDeviceToHost));

    if (loss_csv.is_open()) {
      std::vector<float> loss(iters, 0.0f);
      HIP_API_CHECK(hipMemcpy(loss.data(), d_loss, iters * sizeof(float), hipMemcpyDeviceToHost));
      HIP_API_CHECK(hipDeviceSynchronize());
      for (int it = 0; it < iters; ++it) {
        loss_csv << (it + 1) << "," << loss[it] << "\n";
      }
    }
  }

  void training_summary () {
    int num_correct = 0;
    float final_loss = 0.0f;
    HIP_API_CHECK(hipMemcpy(&num_correct, d_num_correct, sizeof(int), hipMemcpyDeviceToHost));
    HIP_API_CHECK(hipMemcpy(&final_loss, d_final_loss, sizeof(float), hipMemcpyDeviceToHost));
    HIP_API_CHECK(hipDeviceSynchronize());
    std::cout << "\nTraining summary:\n";

    const std::streamsize save_precision = std::cout.precision();
    const std::ios::fmtflags save_flags = std::cout.flags();

    std::cout << std::fixed << std::setprecision(6);
    std::cout << "  Accuracy: " << static_cast<float>(num_correct) / mnist->num_train_images << "\n";
    std::cout << "  Final loss: " << final_loss << "\n";
    std::cout << "  Time to completion: " << training_elapsed_ms / 1000 << " s\n";

    std::cout.flags(save_flags);
    std::cout.precision(save_precision);
  }

  void test () {
    /*
      Run the model on test dataset to output predictions.
    */
    std::cout << "\nTesting model...\n";

    HIP_API_CHECK(hipMemset(d_num_correct, 0, sizeof(int)));

    HIP_KERNEL_CHECK(hipLaunchKernelGGL(
      (sgemm_sub_and_scale<MT, NT, KT>),
      dim3((mnist->k_num_categories + MT - 1) / MT, (mnist->num_test_images + NT - 1) / NT, 1), dim3(MT, NT, 1), 0, 0,
      d_weights, d_x_test, d_y_test, d_yhat_test_scores,
      mnist->k_num_categories, mnist->num_test_images, mnist->pixels_per_image + 1,
      2.0f));
    HIP_KERNEL_CHECK(hipLaunchKernelGGL(calc_num_correct,
                                        dim3(k_num_blocks, 1, 1), dim3(k_threads_per_block, 1, 1), 0, 0,
                                        d_yhat_test_scores, d_y_test, d_yhat_test, d_num_correct,
                                        mnist->k_num_categories, mnist->num_test_images));
    HIP_API_CHECK(hipDeviceSynchronize());

    mnist->yhat_test.resize(mnist->num_test_images, 0);
    HIP_API_CHECK(hipMemcpy(mnist->yhat_test.data(), d_yhat_test, mnist->num_test_images * sizeof(uint8_t), hipMemcpyDeviceToHost));
  }

  void test_summary () {
    int num_correct = 0;
    HIP_API_CHECK(hipMemcpy(&num_correct, d_num_correct, sizeof(int), hipMemcpyDeviceToHost));
    std::cout << "\nTest summary:\n";

    const std::streamsize save_precision = std::cout.precision();
    const std::ios::fmtflags save_flags = std::cout.flags();

    std::cout << std::fixed << std::setprecision(6);

    std::cout << "  Accuracy: " << static_cast<float>(num_correct)/mnist->num_test_images << "\n";

    std::cout.flags(save_flags);
    std::cout.precision(save_precision);
  }
};

int main (int argc, char** argv) {

  Config config(argc, argv);
  MNIST mnist(config);
  Model model(&mnist, config.iters, config.lr);

  model.train<false>("loss.csv");
  model.training_summary();
  mnist.print_image(0, MNIST::DatasetType::TRAIN);

  model.test();
  model.test_summary();
  mnist.print_image(0, MNIST::DatasetType::TEST);

  return 0;
}
