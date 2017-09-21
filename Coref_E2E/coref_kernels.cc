#include <set>
#include <map>

#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/shape_inference.h"
#include "tensorflow/core/framework/op_kernel.h"

using namespace tensorflow;

REGISTER_OP("Spans")
.Attr("max_width: int")
.Input("sentence_indices: int32")
.Output("starts: int32")
.Output("ends: int32");

class SpansOp : public OpKernel {
public:
  explicit SpansOp(OpKernelConstruction* context) : OpKernel(context) {
    OP_REQUIRES_OK(context, context->GetAttr("max_width", &_max_width));
  }

  void Compute(OpKernelContext* context) override {
    TTypes<int32>::ConstVec sentence_indices = context->input(0).vec<int32>();
    int length = sentence_indices.dimension(0);

    std::vector<std::pair<int, int>> spans;
    for (int i = 0; i < length; ++i) {
      for (int j = i; j < length && (j - i) < _max_width; ++j) {
        if (sentence_indices(i) == sentence_indices(j)) {
          spans.emplace_back(i, j);
        }
      }
    }

    Tensor* starts_tensor = nullptr;
    Tensor* ends_tensor = nullptr;
    TensorShape outputs_shape({static_cast<int64>(spans.size())});
    OP_REQUIRES_OK(context, context->allocate_output(0, outputs_shape, &starts_tensor));
    OP_REQUIRES_OK(context, context->allocate_output(1, outputs_shape, &ends_tensor));
    TTypes<int32>::Vec starts = starts_tensor->vec<int32>();
    TTypes<int32>::Vec ends = ends_tensor->vec<int32>();

    for (int i = 0; i < spans.size(); ++i) {
      const std::pair<int, int>& span = spans[i];
      starts(i) = span.first;
      ends(i) = span.second;
    }
  }

private:
  int _max_width;
};

REGISTER_KERNEL_BUILDER(Name("Spans").Device(DEVICE_CPU), SpansOp);


REGISTER_OP("Antecedents")
.Input("mention_starts: int32")
.Input("mention_ends: int32")
.Input("gold_starts: int32")
.Input("gold_ends: int32")
.Input("cluster_ids: int32")
.Input("max_antecedents: int32")
.Output("antecedents: int32")
.Output("antecedent_labels: bool")
.Output("antecedents_len: int32");

class AntecedentsOp : public OpKernel {
public:
  explicit AntecedentsOp(OpKernelConstruction* context) : OpKernel(context) {}

  void Compute(OpKernelContext* context) override {
    TTypes<int32>::ConstVec mention_starts = context->input(0).vec<int32>();
    TTypes<int32>::ConstVec mention_ends = context->input(1).vec<int32>();
    TTypes<int32>::ConstVec gold_starts = context->input(2).vec<int32>();
    TTypes<int32>::ConstVec gold_ends = context->input(3).vec<int32>();
    TTypes<int32>::ConstVec cluster_ids = context->input(4).vec<int32>();

    CHECK_EQ(mention_starts.dimension(0), mention_ends.dimension(0));
    CHECK_EQ(gold_starts.dimension(0), gold_ends.dimension(0));

    int num_mentions = mention_starts.dimension(0);
    int num_gold = gold_starts.dimension(0);

    int max_antecedents = std::min(num_mentions, context->input(5).scalar<int32>()(0));

    Tensor* antecedents_tensor = nullptr;
    TensorShape antecedents_shape({num_mentions, max_antecedents});
    OP_REQUIRES_OK(context, context->allocate_output(0, antecedents_shape, &antecedents_tensor));
    TTypes<int32>::Matrix antecedents = antecedents_tensor->matrix<int32>();

    Tensor* labels_tensor = nullptr;
    TensorShape labels_shape({num_mentions, max_antecedents + 1});
    OP_REQUIRES_OK(context, context->allocate_output(1, labels_shape, &labels_tensor));
    TTypes<bool>::Matrix labels = labels_tensor->matrix<bool>();

    Tensor* antecedents_len_tensor = nullptr;
    TensorShape antecedents_len_shape({num_mentions});
    OP_REQUIRES_OK(context, context->allocate_output(2, antecedents_len_shape, &antecedents_len_tensor));
    TTypes<int32>::Vec antecedents_len = antecedents_len_tensor->vec<int32>();

    std::map<std::pair<int, int>, int> mention_indices;
    for (int i = 0; i < num_mentions; ++i) {
      mention_indices[std::pair<int, int>(mention_starts(i), mention_ends(i))] = i;
    }

    std::vector<int> mention_cluster_ids(num_mentions, -1);
    for (int i = 0; i < num_gold; ++i) {
      auto iter = mention_indices.find(std::pair<int, int>(gold_starts(i), gold_ends(i)));
      if (iter != mention_indices.end()) {
        mention_cluster_ids[iter->second] = cluster_ids(i);
      }
    }

    for (int i = 0; i < num_mentions; ++i) {
      int antecedent_count = 0;
      bool null_label = true;
      for (int j = std::max(0, i - max_antecedents); j < i; ++j) {
        if (mention_cluster_ids[i] >= 0 && mention_cluster_ids[i] == mention_cluster_ids[j]) {
          labels(i, antecedent_count + 1) = true;
          null_label = false;
        } else {
          labels(i, antecedent_count + 1) = false;
        }
        antecedents(i, antecedent_count) = j;
        ++antecedent_count;
      }
      for (int j = antecedent_count; j < max_antecedents; ++j) {
        labels(i, j + 1) = false;
        antecedents(i, j) = 0;
      }
      labels(i, 0) = null_label;
      antecedents_len(i) = antecedent_count;
    }
  }
};
REGISTER_KERNEL_BUILDER(Name("Antecedents").Device(DEVICE_CPU), AntecedentsOp);

REGISTER_OP("ExtractMentions")
.Input("mention_scores: float32")
.Input("candidate_starts: int32")
.Input("candidate_ends: int32")
.Input("num_output_mentions: int32")
.Output("output_mention_indices: int32");

class ExtractMentionsOp : public OpKernel {
public:
  explicit ExtractMentionsOp(OpKernelConstruction* context) : OpKernel(context) {}

  void Compute(OpKernelContext* context) override {
    TTypes<float>::ConstVec mention_scores = context->input(0).vec<float>();
    TTypes<int32>::ConstVec candidate_starts = context->input(1).vec<int32>();
    TTypes<int32>::ConstVec candidate_ends = context->input(2).vec<int32>();
    int num_output_mentions = context->input(3).scalar<int32>()(0);

    int num_input_mentions = mention_scores.dimension(0);

    Tensor* output_mention_indices_tensor = nullptr;
    TensorShape output_mention_indices_shape({num_output_mentions});
    OP_REQUIRES_OK(context, context->allocate_output(0, output_mention_indices_shape, &output_mention_indices_tensor));
    TTypes<int32>::Vec output_mention_indices = output_mention_indices_tensor->vec<int32>();

    std::vector<int> sorted_input_mention_indices(num_input_mentions);
    std::iota(sorted_input_mention_indices.begin(), sorted_input_mention_indices.end(), 0);

    std::sort(sorted_input_mention_indices.begin(), sorted_input_mention_indices.end(),
              [&mention_scores](int i1, int i2) {
                return mention_scores(i2) < mention_scores(i1);
              });
    std::vector<int> top_mention_indices;
    int current_mention_index = 0;
    while (top_mention_indices.size() < num_output_mentions) {
      int i = sorted_input_mention_indices[current_mention_index];
      bool any_crossing = false;
      for (const int j : top_mention_indices) {
        if (is_crossing(candidate_starts, candidate_ends, i, j)) {
          any_crossing = true;
          break;
        }
      }
      if (!any_crossing) {
        top_mention_indices.push_back(i);
      }
      ++current_mention_index;
    }

    std::sort(top_mention_indices.begin(), top_mention_indices.end(),
              [&candidate_starts, &candidate_ends] (int i1, int i2) {
                if (candidate_starts(i1) < candidate_starts(i2)) {
                  return true;
                } else if (candidate_starts(i1) > candidate_starts(i2)) {
                  return false;
                } else if (candidate_ends(i1) < candidate_ends(i2)) {
                  return true;
                } else if (candidate_ends(i1) > candidate_ends(i2)) {
                  return false;
                } else {
                  return i1 < i2;
                }
              });

    for (int i = 0; i < num_output_mentions; ++i) {
      output_mention_indices(i) = top_mention_indices[i];
    }
  }
private:
  bool is_crossing(TTypes<int32>::ConstVec &candidate_starts, TTypes<int32>::ConstVec &candidate_ends, int i1, int i2) {
    int s1 = candidate_starts(i1);
    int s2 = candidate_starts(i2);
    int e1 = candidate_ends(i1);
    int e2 = candidate_ends(i2);
    return (s1 < s2 && s2 <= e1 && e1 < e2) || (s2 < s1 && s1 <= e2 && e2 < e1);
  }
};


REGISTER_KERNEL_BUILDER(Name("ExtractMentions").Device(DEVICE_CPU), ExtractMentionsOp);

REGISTER_OP("DistanceBins")
.Input("distances: int32")
.Output("bins: int32");

class DistanceBinsOp : public OpKernel {
public:
  explicit DistanceBinsOp(OpKernelConstruction* context) : OpKernel(context) {}

  void Compute(OpKernelContext* context) override {
    TTypes<int32>::ConstMatrix distances  = context->input(0).matrix<int32>();

    int d0 = distances.dimension(0);
    int d1 = distances.dimension(1);

    Tensor* bins_tensor = nullptr;
    TensorShape bins_shape({d0, d1});
    OP_REQUIRES_OK(context, context->allocate_output(0, bins_shape, &bins_tensor));
    TTypes<int32>::Matrix bins = bins_tensor->matrix<int32>();

    for (int i = 0; i < d0; ++i) {
      for (int j = 0; j < d1; ++j) {
        bins(i, j) = get_bin(distances(i, j));
      }
    }
  }
private:
  int get_bin(int d) {
    if (d <= 0) {
      return 0;
    } else if (d == 1) {
      return 1;
    } else if (d == 2) {
      return 2;
    } else if (d == 3) {
      return 3;
    } else if (d == 4) {
      return 4;
    } else if (d <= 7) {
      return 5;
    } else if (d <= 15) {
      return 6;
    } else if (d <= 31) {
      return 7;
    } else if (d <= 63) {
      return 8;
    } else {
      return 9;
    }
  }
};


REGISTER_KERNEL_BUILDER(Name("DistanceBins").Device(DEVICE_CPU), DistanceBinsOp);
