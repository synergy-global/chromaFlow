#pragma once

#include "ChromaBaseClasses.h"

namespace ChromaFlow {

class MSEParameterLoss {
public:
    // Mean squared error for parameter vectors
    float calculate(const Eigen::VectorXf& parameter, const Eigen::VectorXf& target) const {
        if (parameter.size() != target.size() || parameter.size() == 0) return 0.0f;
        Eigen::VectorXf diff = parameter - target;
        return diff.squaredNorm() / static_cast<float>(parameter.size());
    }

};
} // namespace ChromaFlow