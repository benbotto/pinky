#include <iostream>
using std::cout;
using std::endl;
#include <array>
using std::array;
#include <exception>
using std::exception;
#include <cmath>
using std::sin;
using std::acos;
#include <random>
using std::uniform_real_distribution;
using std::default_random_engine;
using std::random_device;
#include "NeuralNet.h"
#include "NeuralNetMomento.h"
#include "neuron/Neuron.h"
#include "neuron/InputNeuron.h"
#include "neuron/BiasNeuron.h"

/**
 * Should output a number between 0 and 1, but this function can
 * be changed to learn other functions.
 */
double func(double x) {
  return .5 * (sin(x) + 1);
}

int main(int argc, char* argv[]) {
  try {
    /**
     * Example that learns func().
     */

    const double   pi        = acos(-1);
    const unsigned NUM_TRAIN = 1000;

    busybin::NeuralNet<1, 30, 1> nn;

    // Training data.
    array<double, NUM_TRAIN> tInputs;
    array<double, NUM_TRAIN> tExpected;
    uniform_real_distribution<double>  dist(-pi * 4, pi * 4);
    random_device                      randDev;
    default_random_engine              engine(randDev());

    for (unsigned i = 0; i < NUM_TRAIN; ++i) {
      tInputs[i]   = dist(engine);
      tExpected[i] = func(tInputs[i]);
    }

    // Train until the mean error is sufficiently small.
    double   totalError = 0;
    double   meanError  = 0;
    unsigned iteration  = 0;

    do {
      totalError = 0;
      ++iteration;

      for (unsigned i = 0; i < NUM_TRAIN; ++i)
        totalError += nn.train({tInputs[i]}, {tExpected[i]});

      meanError = totalError / NUM_TRAIN;

      if (iteration % 1000 == 0) {
        cout << "Iteration: "  << iteration
             << " Mean Error: " << meanError
             << endl;
      }
    }
    while (meanError > .0001);

    cout << "Done training.  Mean error: " << meanError
         << " Iterations: " << iteration << endl;

    // Random data.
    for (unsigned i = 0; i < 300; ++i) {
      double x = dist(engine);
      array<double, 1> output = nn.feedForward({x});

      cout << x << "," << output[0] << endl;
    }
  }
  catch (exception& ex) {
    cout << "Error: " << ex.what() << endl;
  }

  return 0;
}

