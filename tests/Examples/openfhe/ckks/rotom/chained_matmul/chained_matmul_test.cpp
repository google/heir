#include <vector>
#include <fstream>
#include <sstream>
#include <string>
#include <iostream>
#include <iomanip>
#include <algorithm>
#include <unordered_set>
#include <cmath>
#include <map>
#include <dirent.h>

#include "gtest/gtest.h"  // from @googletest

// Generated headers (block clang-format from messing up order)
#include "tests/Examples/openfhe/ckks/rotom/chained_matmul/chained_matmul_lib.h"

namespace mlir {
namespace heir {
namespace openfhe {

// Reads all .txt files from a directory and maps filename to vector of doubles.
// File format: first line contains vector size, then one value per line.
// Returns a map from filename (with extension) to vector of doubles.
std::map<std::string, std::vector<double>> readFromDirectory(const std::string& dirpath) {
  std::map<std::string, std::vector<double>> result;
  
  DIR* dir = opendir(dirpath.c_str());
  if (dir == nullptr) {
    std::cerr << "Error: Could not open directory " << dirpath << std::endl;
    return result;
  }
  
  struct dirent* entry;
  while ((entry = readdir(dir)) != nullptr) {
    std::string filename = entry->d_name;
    
    if (filename == "." || filename == "..") {
      continue;
    }
    
    if (filename.size() < 4 || filename.substr(filename.size() - 4) != ".txt") {
      continue;
    }
    
    std::string fullpath = dirpath + "/" + filename;
    std::ifstream file(fullpath);
    if (!file.is_open()) {
      std::cerr << "Warning: Could not open file " << fullpath << std::endl;
      continue;
    }
    
    std::string line;
    int vector_size = 0;
    
    if (std::getline(file, line)) {
      std::istringstream iss(line);
      if (!(iss >> vector_size)) {
        std::cerr << "Warning: Could not parse vector size from " << fullpath << std::endl;
        file.close();
        continue;
      }
    }
    
    std::vector<double> values;
    while (std::getline(file, line) && values.size() < static_cast<size_t>(vector_size)) {
      std::istringstream iss(line);
      double value;
      if (iss >> value) {
        values.push_back(value);
      }
    }
    
    file.close();
    
    if (values.size() != static_cast<size_t>(vector_size)) {
      std::cerr << "Warning: Expected " << vector_size << " values but got " << values.size() 
                << " in " << fullpath << std::endl;
    }
    
    result[filename] = values;
  }
  
  closedir(dir);
  return result;
}

// Extracts and sorts input filenames numerically, excluding specified filenames.
// Returns a sorted vector of filenames based on the numeric value in the filename.
std::vector<std::string> getSortedInputFilenames(
    const std::map<std::string, std::vector<double>>& inputs_map,
    const std::unordered_set<std::string>& exclude_filenames = {}) {
  std::vector<std::string> input_filenames;
  
  for (const auto& [filename, vec] : inputs_map) {
    if (exclude_filenames.find(filename) == exclude_filenames.end()) {
      input_filenames.push_back(filename);
    }
  }
  
  std::sort(input_filenames.begin(), input_filenames.end(), [](const std::string& a, const std::string& b) {
    try {
      std::string a_num = a.substr(0, a.find('.'));
      std::string b_num = b.substr(0, b.find('.'));
      return std::stoi(a_num) < std::stoi(b_num);
    } catch (...) {
      return a < b;
    }
  });
  
  return input_filenames;
}


// Tests chained matrix multiplication, a @ b @ c,
// where a is an encrypted input matrix, and b and c are plaintext matrices.
// All matrices have the same shape (64x64), and n=4096.
TEST(ChainedMatmulTest, OrderIndependentResultCheck) {
  // Initialize crypto context
  auto cryptoContext = chained_matmul__generate_crypto_context();
  auto keyPair = cryptoContext->KeyGen();
  cryptoContext = chained_matmul__configure_crypto_context(cryptoContext, keyPair.secretKey);

  // Read inputs from inputs/ directory and map filename to vector
  auto inputs_map = readFromDirectory("tests/Examples/openfhe/ckks/rotom/chained_matmul/inputs");  
  // Read result from results/ directory
  auto expected_result = readFromDirectory("tests/Examples/openfhe/ckks/rotom/chained_matmul/results");

  // Encrypt the specified input as the secret argument
  const std::string encrypted_input = "2.txt";
  // Extract and sort input filenames numerically, excluding the one used for encryption
  std::vector<std::string> input_filenames = getSortedInputFilenames(
      inputs_map, {encrypted_input});
  
  // Check we have enough inputs (function takes 128 plaintext arguments)
  const size_t num_args = 128;
  ASSERT_GE(input_filenames.size(), num_args) 
    << "Need at least " << num_args << " input files (excluding encrypted input)";

  auto ctEncrypted = chained_matmul__encrypt__arg0(cryptoContext, inputs_map[encrypted_input], keyPair.publicKey);

  auto result = chained_matmul(cryptoContext, ctEncrypted,
    inputs_map[input_filenames[0]], inputs_map[input_filenames[1]], inputs_map[input_filenames[2]], inputs_map[input_filenames[3]],
    inputs_map[input_filenames[4]], inputs_map[input_filenames[5]], inputs_map[input_filenames[6]], inputs_map[input_filenames[7]],
    inputs_map[input_filenames[8]], inputs_map[input_filenames[9]], inputs_map[input_filenames[10]], inputs_map[input_filenames[11]],
    inputs_map[input_filenames[12]], inputs_map[input_filenames[13]], inputs_map[input_filenames[14]], inputs_map[input_filenames[15]],
    inputs_map[input_filenames[16]], inputs_map[input_filenames[17]], inputs_map[input_filenames[18]], inputs_map[input_filenames[19]],
    inputs_map[input_filenames[20]], inputs_map[input_filenames[21]], inputs_map[input_filenames[22]], inputs_map[input_filenames[23]],
    inputs_map[input_filenames[24]], inputs_map[input_filenames[25]], inputs_map[input_filenames[26]], inputs_map[input_filenames[27]],
    inputs_map[input_filenames[28]], inputs_map[input_filenames[29]], inputs_map[input_filenames[30]], inputs_map[input_filenames[31]],
    inputs_map[input_filenames[32]], inputs_map[input_filenames[33]], inputs_map[input_filenames[34]], inputs_map[input_filenames[35]],
    inputs_map[input_filenames[36]], inputs_map[input_filenames[37]], inputs_map[input_filenames[38]], inputs_map[input_filenames[39]],
    inputs_map[input_filenames[40]], inputs_map[input_filenames[41]], inputs_map[input_filenames[42]], inputs_map[input_filenames[43]],
    inputs_map[input_filenames[44]], inputs_map[input_filenames[45]], inputs_map[input_filenames[46]], inputs_map[input_filenames[47]],
    inputs_map[input_filenames[48]], inputs_map[input_filenames[49]], inputs_map[input_filenames[50]], inputs_map[input_filenames[51]],
    inputs_map[input_filenames[52]], inputs_map[input_filenames[53]], inputs_map[input_filenames[54]], inputs_map[input_filenames[55]],
    inputs_map[input_filenames[56]], inputs_map[input_filenames[57]], inputs_map[input_filenames[58]], inputs_map[input_filenames[59]],
    inputs_map[input_filenames[60]], inputs_map[input_filenames[61]], inputs_map[input_filenames[62]], inputs_map[input_filenames[63]],
    inputs_map[input_filenames[64]], inputs_map[input_filenames[65]], inputs_map[input_filenames[66]], inputs_map[input_filenames[67]],
    inputs_map[input_filenames[68]], inputs_map[input_filenames[69]], inputs_map[input_filenames[70]], inputs_map[input_filenames[71]],
    inputs_map[input_filenames[72]], inputs_map[input_filenames[73]], inputs_map[input_filenames[74]], inputs_map[input_filenames[75]],
    inputs_map[input_filenames[76]], inputs_map[input_filenames[77]], inputs_map[input_filenames[78]], inputs_map[input_filenames[79]],
    inputs_map[input_filenames[80]], inputs_map[input_filenames[81]], inputs_map[input_filenames[82]], inputs_map[input_filenames[83]],
    inputs_map[input_filenames[84]], inputs_map[input_filenames[85]], inputs_map[input_filenames[86]], inputs_map[input_filenames[87]],
    inputs_map[input_filenames[88]], inputs_map[input_filenames[89]], inputs_map[input_filenames[90]], inputs_map[input_filenames[91]],
    inputs_map[input_filenames[92]], inputs_map[input_filenames[93]], inputs_map[input_filenames[94]], inputs_map[input_filenames[95]],
    inputs_map[input_filenames[96]], inputs_map[input_filenames[97]], inputs_map[input_filenames[98]], inputs_map[input_filenames[99]],
    inputs_map[input_filenames[100]], inputs_map[input_filenames[101]], inputs_map[input_filenames[102]], inputs_map[input_filenames[103]],
    inputs_map[input_filenames[104]], inputs_map[input_filenames[105]], inputs_map[input_filenames[106]], inputs_map[input_filenames[107]],
    inputs_map[input_filenames[108]], inputs_map[input_filenames[109]], inputs_map[input_filenames[110]], inputs_map[input_filenames[111]],
    inputs_map[input_filenames[112]], inputs_map[input_filenames[113]], inputs_map[input_filenames[114]], inputs_map[input_filenames[115]],
    inputs_map[input_filenames[116]], inputs_map[input_filenames[117]], inputs_map[input_filenames[118]], inputs_map[input_filenames[119]],
    inputs_map[input_filenames[120]], inputs_map[input_filenames[121]], inputs_map[input_filenames[122]], inputs_map[input_filenames[123]],
    inputs_map[input_filenames[124]], inputs_map[input_filenames[125]], inputs_map[input_filenames[126]], inputs_map[input_filenames[127]]);

  auto actual = chained_matmul__decrypt__result0(cryptoContext, result, keyPair.secretKey);


  ASSERT_GT(actual.size(), 0) << "Actual result should not be empty";
  ASSERT_GT(expected_result["result_0.txt"].size(), 0) << "Expected result should not be empty";

  // Check that the actual result is close to the expected result
  for (size_t i = 0; i < actual.size(); ++i) {
    EXPECT_NEAR(expected_result["result_0.txt"][i], actual[i], 1e-3);
  }
}

}  // namespace openfhe
}  // namespace heir
}  // namespace mlir