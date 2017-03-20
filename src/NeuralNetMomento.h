#ifndef _BUSYBIN_NEURAL_NET_MOMENTO_H_
#define _BUSYBIN_NEURAL_NET_MOMENTO_H_

#include <array>
using std::array;
#include <fstream>
using std::ifstream;
using std::ofstream;
#include <exception>
using std::runtime_error;
#include "NeuralNet.h"
using busybin::NeuralNet;

namespace busybin {
  template <size_t NUM_IN, size_t NUM_HIDDEN, size_t NUM_OUT>
  class NeuralNetMomento {
    typedef array<double, (NUM_IN + 1) * NUM_HIDDEN + (NUM_HIDDEN + 1) * NUM_OUT> weight_arr_t;
  public:
    /**
     * Save the NeuralNet's weights to a file.  The weights are saved in a
     * binary format, which means that the resulting file will differ based on
     * architecture.  (E.g. a file saved under IA_64 probably won't load
     * correctly on ARM.)
     */
    void save(const NeuralNet<NUM_IN, NUM_HIDDEN, NUM_OUT>& nn,
      const string& fileName) const {

      ofstream ofile(fileName, std::ios::binary);

      if (!ofile.is_open())
        throw runtime_error("Failed to open weights file.");

      // Write the weights one by one.
      for (double weight : nn.getWeights()) {
        ofile.write(reinterpret_cast<const char*>(&weight), sizeof(double));

        if (ofile.bad())
          throw runtime_error("Failed to write weight.");
      }
      ofile.close();
    }

    /**
     * Load the weights from a file into an array.
     */
    weight_arr_t loadWeights(const string& fileName) const {
      weight_arr_t  weights;
      ifstream      ifile(fileName, std::ios::binary);

      if (!ifile.is_open())
        throw runtime_error("Failed to open weights file.");

      // Load the weights.
      for (unsigned i = 0; i < weights.size(); ++i) {
        ifile.read(reinterpret_cast<char*>(&weights[i]), sizeof(double));

        // Possible size mismatch or other failure.
        if (ifile.bad() || ifile.eof())
          throw runtime_error("Failed to read weight.");
      }

      ifile.close();

      return weights;
    }

    /**
     * Create a NeuralNet instance using the weights from a file.
     */
    NeuralNet<NUM_IN, NUM_HIDDEN, NUM_OUT> load(const string& fileName) const {
      return NeuralNet<NUM_IN, NUM_HIDDEN, NUM_OUT>(this->loadWeights(fileName));
    }
  };
}

#endif

