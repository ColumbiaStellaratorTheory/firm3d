#pragma once

#include <vector>

// Shared RHS interface, independent of the magnetic-field implementations.
class BaseRHS {
public:
    virtual ~BaseRHS() = default;
    virtual void operator()(const std::vector<double>& y,
                            std::vector<double>& dydt, double t) = 0;
    virtual int get_state_size() const = 0;
};
