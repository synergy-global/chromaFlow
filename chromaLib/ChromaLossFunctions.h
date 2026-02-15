#pragma once

#include "ChromaBaseClasses.h"

namespace ChromaFlow::LossFunctions
{

    
static inline Eigen::VectorXf getRow0Safe(const FeatureTensor& t, int N)
{
    Eigen::VectorXf v = Eigen::VectorXf::Zero(N);

    if (t.data.rows() > 0)
    {
        int T = static_cast<int>(t.data.cols());
        int n = std::min(T, N);
        v.head(n) = t.data.row(0).transpose().head(n);
    }

    return v;
}

static inline FeatureTensor makeGrad(const Eigen::VectorXf& g)
{
    FeatureTensor out;
    out.data.resize(1, g.size());
    out.data.row(0) = g.transpose();
    out.numSamples = 1;
    out.features   = static_cast<int>(g.size());
    return out;
}


class MSELoss
{
public:
    FeatureTensor calculate(const FeatureTensor& output,
                            const FeatureTensor& target) const
    {
        const int N = static_cast<int>(output.data.cols());
        if (N <= 0) return makeGrad(Eigen::VectorXf());

        Eigen::VectorXf y = getRow0Safe(output, N);
        Eigen::VectorXf t = getRow0Safe(target, N);

        Eigen::VectorXf grad = (2.0f / static_cast<float>(N)) * (y - t);
        return makeGrad(grad);
    }
};


class L1Loss
{
public:
    FeatureTensor calculate(const FeatureTensor& output,
                            const FeatureTensor& target) const
    {
        const int N = static_cast<int>(output.data.cols());
        if (N <= 0) return makeGrad(Eigen::VectorXf());

        Eigen::VectorXf y = getRow0Safe(output, N);
        Eigen::VectorXf t = getRow0Safe(target, N);

        Eigen::VectorXf grad = (y - t).array().sign();
        grad /= static_cast<float>(N);

        return makeGrad(grad);
    }
};


class EnergyHomeostasisLoss
{
public:
    FeatureTensor calculate(const FeatureTensor& output,
                            float targetEnergy) const
    {
        const int N = static_cast<int>(output.data.cols());
        if (N <= 0) return makeGrad(Eigen::VectorXf());

        Eigen::VectorXf y = getRow0Safe(output, N);

        float energy = y.squaredNorm() / static_cast<float>(N);
        float error  = energy - targetEnergy;

        Eigen::VectorXf grad = (2.0f / static_cast<float>(N)) * error * y;

        return makeGrad(grad);
    }
};


class VarianceLoss
{
public:
    FeatureTensor calculate(const FeatureTensor& output,
                            float targetVar) const
    {
        const int N = static_cast<int>(output.data.cols());
        if (N <= 0) return makeGrad(Eigen::VectorXf());

        Eigen::VectorXf y = getRow0Safe(output, N);

        float mean = y.mean();
        Eigen::VectorXf centered = y.array() - mean;

        float var = centered.squaredNorm() / static_cast<float>(N);
        float error = var - targetVar;

        Eigen::VectorXf grad =
            (2.0f / static_cast<float>(N)) * error * centered;

        return makeGrad(grad);
    }
};
 

class PredictiveLoss
{
public:
    FeatureTensor calculate(const FeatureTensor& prediction,
                            const FeatureTensor& nextFrame) const
    {
        const int N = static_cast<int>(prediction.data.cols());
        if (N <= 0) return makeGrad(Eigen::VectorXf());

        Eigen::VectorXf y = getRow0Safe(prediction, N);
        Eigen::VectorXf t = getRow0Safe(nextFrame, N);

        Eigen::VectorXf grad = (y - t) / static_cast<float>(N);

        return makeGrad(grad);
    }
};

} // namespace ChromaFlow