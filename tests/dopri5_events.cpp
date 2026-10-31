#include "dopri5_dense_output.h"
#include "dopri5_events.h"
#include <iostream>
#include <stdexcept>

using namespace firm3d_events;
void check(bool condition, const char* message) {
    if(!condition) throw std::runtime_error(message);
}
void close(double a, double b, double tolerance = 2e-12) {
    if(std::abs(a-b) > tolerance){ std::cerr << "actual=" << a << " expected=" << b << " difference=" << a-b << "\n"; throw std::runtime_error("Dense event reference mismatch"); }
}
int main() {
    double k[] = {1.2, -0.3, 2.1, 0.7, -1.1, 1.8};
    Polynomial dense = dense_polynomial(0.7, 0.3, k);
    for(int j=0; j<=100; ++j){
        double q = j/100.0;
        close(value(dense,q), dopri5_dense_state(0.7,0.3,q,k[0],k[1],k[2],k[3],k[4],k[5]));
    }
    Polynomial cubic; cubic.degree=3;
    // (q-.2)(q-.5)(q-.8): three roots, despite equal endpoint signs on subintervals.
    cubic.c[0]=-.08; cubic.c[1]=.66; cubic.c[2]=-1.5; cubic.c[3]=1;
    double zeros[max_degree+1]; int count=roots(cubic,0,1,zeros);
    check(count==3,"Lost a polynomial root");
    for(int j=0; j<3; ++j) close(zeros[j],.2+.3*j);
    Polynomial tangent; tangent.degree=2; tangent.c[0]=.25; tangent.c[1]=-1; tangent.c[2]=1;
    check(roots(tangent,0,1,zeros)==1,"Stationary root missing");
    check(crossing_direction(tangent,zeros[0])==0,"Tangency classified as a crossing");
    Step step{}; step.t=0; step.h=1; step.theta_offset=step.zeta_offset=0;
    step.state[0].c[0]=1;
    step.state[2].degree=1; step.state[2].c[1]=20*period;
    double plane[]={0,1,0,0};
    // The CPU selects only the next periodic plane from the step endpoints.
    close(phase_crossing(step,plane,1),.05);
    step.state[2].c[1]=-20*period;
    close(phase_crossing(step,plane,1),.05);
    // Opposite crossings without an endpoint bracket are omitted, as on CPU.
    step.state[2].degree=2; step.state[2].c[0]=-.25;
    step.state[2].c[1]=2;step.state[2].c[2]=-2;
    check(phase_crossing(step,plane,1)>1,"Unbracketed phase crossing emitted");
    close(phase_crossing(step,plane,.4),(1-std::sqrt(.5))/2);
    Polynomial velocity; velocity.degree=2;
    velocity.c[0]=.1875; velocity.c[1]=-1; velocity.c[2]=1;
    check(velocity_crossing(velocity,0,1)>1,"Unbracketed velocity crossing emitted");
    close(velocity_crossing(velocity,0,.5),.25);
    velocity.c[0]=0;
    check(velocity_crossing(velocity,0,1)>1,"Launch velocity crossing emitted");
    velocity.c[0]=1; velocity.c[1]=-2; // (q-1)^2 touches the right endpoint.
    close(velocity_crossing(velocity,0,1),1);
    // A helical plane crosses the atan2 branch without an artificial jump.
    step.state[0].c[0]=-1;step.state[1].degree=1;
    step.state[1].c[0]=.1;step.state[1].c[1]=-.2;
    step.nbranches=roots(step.state[1],0,1,step.branch_roots);
    double helical[]={period/2,0,1,0};
    close(step.theta(.5),period/2);
    check(step.theta(.75)>period/2,"Angle winding lost");
    close(phase_crossing(step,helical,1),.5);
    // Time-dependent phase and clipped terminal time.
    double timed[]={0,0,0,-period};
    close(phase_crossing(step,timed,1),1);
    check(phase_crossing(step,timed,.9)>.9,"Crossing beyond tmax emitted");
    std::cout << "Dense event polynomial, roots, winding, and phase checks passed.\n";
}
