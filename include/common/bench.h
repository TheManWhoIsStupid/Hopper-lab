#pragma once

#include <fstream>
#include <string>
#include <vector>

namespace hopper {

// 传输 bytes 字节用时 ms，换算带宽 GB/s
inline double bandwidth_gbps(double bytes, double ms) {
  return bytes / (ms * 1e6);
}

// 共 flops 次浮点运算用时 ms，换算 TFLOPS
inline double tflops(double flops, double ms) { return flops / (ms * 1e9); }

// 极简 CSV 记录器：文件不存在（或为空）时写表头，之后追加行。
// 用途：实验数据统一落盘 results/，方便后续画图对比。
class CsvWriter {
 public:
  CsvWriter(std::string path, std::vector<std::string> columns)
      : path_(std::move(path)), columns_(std::move(columns)) {
    bool need_header = true;
    {
      std::ifstream in(path_);
      if (in.good() && in.peek() != EOF) need_header = false;
    }
    out_.open(path_, std::ios::app);
    if (need_header) write_row(columns_);
  }

  void row(const std::vector<std::string>& values) { write_row(values); }

 private:
  void write_row(const std::vector<std::string>& values) {
    for (size_t i = 0; i < values.size(); ++i) {
      if (i) out_ << ',';
      out_ << values[i];
    }
    out_ << '\n';
    out_.flush();
  }

  std::string path_;
  std::vector<std::string> columns_;
  std::ofstream out_;
};

}  // namespace hopper
