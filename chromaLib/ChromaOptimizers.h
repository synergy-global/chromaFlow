#pragma once

#include "ChromaBaseClasses.h"

namespace ChromaFlow {

/**
 * The baseline optimizer, using a classic momentum update rule for smooth learning.
 */
class SGDWithMomentum : public IOptimizer {
public:
    SGDWithMomentum(float learningRate = 0.001f, float momentum = 0.9f)
        : learningRate_(learningRate), momentumCoeff_(momentum) {}
        
    Eigen::VectorXf update(const Eigen::VectorXf& parameter,
                             const Eigen::VectorXf& gradient,
                             const FeatureTensor& features) override {
        // Initialize momentum buffer if it's not the right shape
        if (momentum_.size() != gradient.size()) {
            momentum_.resize(gradient.size());
            momentum_.setZero();
        }
        
        // Classic momentum update rule
        for (size_t i = 0; i < gradient.size(); ++i) {
            momentum_[i] = momentumCoeff_ * momentum_[i] + (1.0f - momentumCoeff_) * gradient[i];
        }
        
        // Update parameters
        std::vector<float> updatedParameter(parameter.size());
        for (size_t i = 0; i < parameter.size(); ++i) {
            updatedParameter[i] = parameter[i] - learningRate_ * momentum_[i];
        }
        
        return Eigen::Map<Eigen::VectorXf>(updatedParameter.data(), updatedParameter.size());
    }

    
    void update(Eigen::VectorXf& parameter,
                const Eigen::VectorXf& gradient,
                const FeatureTensor& features) override {
        if (momentum_.size() != gradient.size()) {
            momentum_.resize(gradient.size());
            momentum_.setZero();
        }
        momentum_ = momentumCoeff_ * momentum_ + (1.0f - momentumCoeff_) * gradient;
        parameter.noalias() -= learningRate_ * momentum_;
    }
    void update(Eigen::MatrixXf& parameter,
                const Eigen::MatrixXf& gradient,
                const FeatureTensor& features) override {
        if (momentumM_.rows() != gradient.rows() || momentumM_.cols() != gradient.cols()) {
            momentumM_.resize(gradient.rows(), gradient.cols());
            momentumM_.setZero();
        }
        momentumM_ = momentumCoeff_ * momentumM_ + (1.0f - momentumCoeff_) * gradient;
        parameter.noalias() -= learningRate_ * momentumM_;
    }
    void reset() {
        momentum_.setZero();
        momentumM_.setZero();
    }
    
private:
    float learningRate_;
    float momentumCoeff_;
    Eigen::VectorXf momentum_;
    Eigen::MatrixXf momentumM_;

};

/**
 * Adam optimizer implementation for more sophisticated gradient-based optimization.
 */
class AdamOptimizer : public IOptimizer {
public:
    AdamOptimizer(float learningRate = 0.001f, float beta1 = 0.9f, float beta2 = 0.999f, float epsilon = 1e-8f)
        : learningRate_(learningRate), beta1_(beta1), beta2_(beta2), epsilon_(epsilon), t_(0) {}
    
    Eigen::VectorXf update(const Eigen::VectorXf& parameter,
                             const Eigen::VectorXf& gradient,
                             const FeatureTensor& features) override {
        if (m_.size() != gradient.size()) {
            m_.resize(gradient.size());
            v_.resize(gradient.size());
            m_.setZero();
            v_.setZero();
        }
        t_++;
        m_ = beta1_ * m_ + (1.0f - beta1_) * gradient;
        v_ = beta2_ * v_ + (1.0f - beta2_) * gradient.cwiseProduct(gradient);
        const float beta1_t = std::pow(beta1_, t_);
        const float beta2_t = std::pow(beta2_, t_);
        Eigen::VectorXf m_hat = m_ / (1.0f - beta1_t);
        Eigen::VectorXf v_hat = v_ / (1.0f - beta2_t);
        Eigen::VectorXf updated = parameter - learningRate_ * m_hat.cwiseQuotient(v_hat.array().sqrt().matrix() + Eigen::VectorXf::Constant(v_hat.size(), epsilon_));
        return updated;
    }
    
    void update(Eigen::VectorXf& parameter,
                const Eigen::VectorXf& gradient,
                const FeatureTensor& features) override {
        if (m_.size() != gradient.size()) {
            m_.resize(gradient.size());
            v_.resize(gradient.size());
            m_.setZero();
            v_.setZero();
        }
        t_++;
        m_ = beta1_ * m_ + (1.0f - beta1_) * gradient;
        v_ = beta2_ * v_ + (1.0f - beta2_) * gradient.cwiseProduct(gradient);
        const float beta1_t = std::pow(beta1_, t_);
        const float beta2_t = std::pow(beta2_, t_);
        Eigen::VectorXf m_hat = m_ / (1.0f - beta1_t);
        Eigen::VectorXf v_hat = v_ / (1.0f - beta2_t);
        parameter.noalias() -= learningRate_ * m_hat.cwiseQuotient(v_hat.array().sqrt().matrix() + Eigen::VectorXf::Constant(v_hat.size(), epsilon_));
    }
    // ... existing code ...
    // New in-place matrix update overload
    void update(Eigen::MatrixXf& parameter,
                const Eigen::MatrixXf& gradient,
                const FeatureTensor& features) override {
        if (mM_.rows() != gradient.rows() || mM_.cols() != gradient.cols()) {
            mM_.resize(gradient.rows(), gradient.cols());
            vM_.resize(gradient.rows(), gradient.cols());
            mM_.setZero();
            vM_.setZero();
        }
        t_++;
        mM_ = beta1_ * mM_ + (1.0f - beta1_) * gradient;
        vM_ = beta2_ * vM_ + (1.0f - beta2_) * gradient.cwiseProduct(gradient);
        const float beta1_t = std::pow(beta1_, t_);
        const float beta2_t = std::pow(beta2_, t_);
        Eigen::MatrixXf m_hat = mM_ / (1.0f - beta1_t);
        Eigen::MatrixXf v_hat = vM_ / (1.0f - beta2_t);
        parameter.noalias() -= learningRate_ * m_hat.cwiseQuotient((v_hat.array().sqrt() + epsilon_).matrix());
    }
    // ... existing code ...
    void reset() {
        m_.setZero();
        v_.setZero();
        mM_.setZero();
        vM_.setZero();
        t_ = 0;
    }
private:
    float learningRate_;
    float beta1_;
    float beta2_;
    float epsilon_;
    int t_; // Time step
    Eigen::VectorXf m_; // First moment estimate
    Eigen::VectorXf v_; // Second moment estimate
    Eigen::MatrixXf mM_; // First moment estimate (matrix)
    Eigen::MatrixXf vM_; // Second moment estimate (matrix)
};
} // namespace ChromaFlow