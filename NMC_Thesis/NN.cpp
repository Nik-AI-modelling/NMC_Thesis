#pragma once

#include "matrix.h"
#include "NN.h"
#include "layer.h"
#include <iostream>
#include <string>
#include <cassert>
#include <fstream>
#include <sstream>

#include <vector>



NN::NN() {}

float NN:: error(Matrix& out, Matrix& exp) {
	return (out - exp).square().sum() / out.cols();
}


//Constractor for NN

//std::vector<int> config = {3, 5, 2};  // 3 neurons (input), 5 neurons (hidden), 2 neurons (output)
//std::vector<std::string> labels = { "A", "B" };
//Input layer: 3 neurons.
//Hidden layer : 5 neurons.
//Output layer : 2 neurons(matches the size of labels).
//NN neural_network(config, labels);

//: output_labels(output_labels) → Member initialization list, which initializes the class member output_labels.


NN::NN(const std::vector<int>& config, const std::vector<std::string>& output_labels)
    : output_labels(output_labels) {

    assert(config.size() >= 1);
    assert(output_labels.size() == config.at(config.size() - 1));

    for (size_t i = 0; i < config.size(); i++) {
        
        int neurons_count = config[i]; //this counts number of neurons in each layer
        if (i == 0) { //this is the input layer
            //The first layer is initialized as a standalone input layer.
            //std::vector<Layer> layers;
            layers.push_back(Layer(neurons_count));
        }
        else {

            //Matrix& Matrix::init(int rows, int cols, matrix_t val) {
              //  this->_rows = rows;
              //  this->_cols = cols;
              //  this->_data = std::vector<matrix_t>(rows * cols, val);


            Layer& prev = layers.at(layers.size() - 1);
            //prev.next_layer(neurons_count) → Creates a new layer connected to the previous one.
            layers.push_back(prev.next_layer(neurons_count));
        }
    }

    //Layer& layer:

  //  Layer is assumed to be a struct.
  //      layer & means we are using a reference, which prevents unnecessary copying of Layer objects, making the code more efficient.
  //      Each iteration of the loop assigns a reference to a Layer object in layers to layer.
  //      : layers:

  //  layers is assumed to be a container(like std::vector<Layer>).
  //      The loop iterates over each element in layers.
  //      layer.weights.randomize(-.5, .5); :

  //      Each Layer object has a weights member, which is assumed to be an instance of a class that has a randomize() method.
  //      The randomize(-0.5, 0.5) function sets the weights of the Layer to random values between - 0.5 and 0.5.


    for (Layer& layer : layers) {
        layer.weights.randomize(-.5, .5);
    }
}


//This function belongs to a class called NN (presumably a Neural Network class) 
//and is used to retrieve the outputs of the last layer in the network.
Matrix& NN::get_outputs() {
    return layers[layers.size() - 1].outputs;
}

//responsible for performing a forward pass through the network.
//input is a reference to a Matrix, ensuring no unnecessary copying.
//The const keyword prevents modifications to input.
void NN::forward(const Matrix& input) {
    //The first layer's outputs (layers[0].outputs) are set equal to input, 
    //meaning the input data is directly assigned as the output of the first layer.
    layers[0].outputs = input;
    for (size_t i = 1; i < layers.size(); i++) {
        Layer& curr = layers[i];
        Layer& prev = layers[i - 1];
        //This is to clarify what forward means
        //void Layer::forward(Layer& curr, Layer& prev) {
        //curr.outputs = (
          //  (prev.outputs * prev.weights) += curr.biased
           // ).sigmoid();
    //}
        Layer::forward(curr, prev);
    }
}

//The Expected matrix contains the expected output of the network. It's also called training data
//. We’ll compare the output of our model with the expected output and tune the parameters (weights and biased)
//of the neural network in a way that it’ll minimize the error.
void NN::backprop(const Matrix& expected) {

    //Remeber , we can only modify weights and biases

//dy/dx is simply how much of y changes for a single unit of change in x at a certain point of x.
//So, we’ll divide our error accordingly to that impact and distribute to all the parameters and 
//we repeat this from the last layer to the first layer and that process is called backward propagation.

    //Output matrix from the last matrix 
    Matrix& output = layers[layers.size() - 1].outputs;
    //
    assert(expected.rows() == output.rows() &&
        expected.cols() == output.cols());

    // delta_out = out - exp
    // delta_hidden = w.trans() * next_delta x (a * (1-a))
    //
    // curr_b += -learn_rate * curr_delta
    // prev_w += -learn_rate * (curr_delta.trans() * prev_active)

    Matrix delta = output - expected;

    //This moves from the last layer to the first one

    //Transpose is used in order to match dimensions of the matrix
    for (size_t i = layers.size() - 1; i > 0; i--) {
        Layer& curr = layers[i];
        Layer& prev = layers[i - 1];
        //This shows how much to move and where to move to minimise the loss function
        //Bias is updated using gradient descent bnew=bold-learn_rate*delta
        //delta * (-learn_rate): Gradient of the bias term scaled by the learning rate.
        curr.biased += (delta * (-learn_rate));

        prev.weights += (prev.outputs.transpose() * delta) * (-learn_rate);

        // sigmoid_derivative = (a * (1 - a));
        Matrix one = Matrix(prev.outputs.rows(), prev.outputs.cols(), 1);
        Matrix sigmoid_derivative = prev.outputs.multiply(one - prev.outputs);

        // delta_next = (delta * prev.w.trans()) x (a * (1-a));
        delta = (delta * prev.weights.transpose()).multiply_inplace(sigmoid_derivative);
    }
}

static void write_matrix(std::ofstream& file, const Matrix& m) {
    //Retrives the number of rows and columns of the matrix m
    int rows = m.rows(), cols = m.cols();
    //Ensures that the number of elements in the matrix is exactly rows * cols.
    assert(m.data().size() == rows * cols);
    //Converts the rows and cols integers into raw bytes ((const char*)&rows) and writes them to the file.
    file.write((const char*)&rows, sizeof rows);
    file.write((const char*)&cols, sizeof cols);


    for (matrix_t val : m.data()) {
        file.write((const char*)&val, sizeof val);
    }
}

static Matrix read_matrix(std::ifstream& file) {
    int rows, cols;
    file.read((char*)&rows, sizeof rows);
    file.read((char*)&cols, sizeof cols);
    assert(rows >= 0 && cols >= 0);

    Matrix m(rows, cols);
    std::vector<matrix_t>& data = m.data();
    for (size_t i = 0; i < rows * cols; i++) {
        matrix_t val;
        file.read((char*)&val, sizeof val);
        data[i] = val;
    }

    return m;
}


//const char* path → A C-style string representing the file path where the neural network data will be saved.
void NN::save(const char* path) const {

    //std::ofstream → This creates an output file stream, meaning it will be used to write data to a file.
    //std::ios::binary → The file is opened in binary mode,
    //meaning data will be written as raw bytes rather than formatted text.
    std::ofstream file(path, std::ios::binary);

    //!!file → This converts file into a boolean (true if the file opened successfully, false if not).
    assert(!!file);
    //(const char*)(&trained) → Converts the address of trained to a char*, which is required by file.write().
     //sizeof trained → Ensures the correct number of bytes are written.
    file.write((const char*)(&trained), sizeof trained);
    file.write((const char*)(&data_index), sizeof data_index);


    //layers.size() → Gets the number of layers in the neural network.
    int layer_count = (int)layers.size();
    //Writes layer_count to the file.
    file.write((const char*)(&layer_count), sizeof layer_count);


    //Loops through each Layer in the layers collection.
    for (const Layer& layer : layers) {

        //Gets the number of columns in the outputs matrix (which represents neurons in the layer).
        int activation_count = (int)layer.outputs.cols();

        //outputs has one row.
        //biased has one row.
        //The number of neurons(columns in outputs) matches the number of bias terms(columns in biased).

        assert(
            layer.outputs.rows() == 1 &&
            layer.biased.rows() == 1 &&
            activation_count == layer.biased.cols()
        );


        //write_matrix(file, layer.biased); → Saves the bias matrix.
        write_matrix(file, layer.biased);
        write_matrix(file, layer.weights);
    }

    file.close();
}

// Function to clean and convert a string to double
std::string cleanAndConvertToDouble(const std::string& str) {
    std::string cleanStr;

    // Keep only digits, '.', and '-'
    for (char ch : str) {
        if (std::isdigit(ch) || ch == '.') { // || ch == '-') {
            cleanStr += ch;
        }
    }

    // Convert to double if valid
    return !cleanStr.empty() ? cleanStr : "";
        //std::stod(cleanStr) : ;
   // condition ? expression_if_true : expression_if_false;
}


void NN::load(const std::string pathX, const std::string pathY, const std::string pathVelx, const std::string pathVely) {
    std::vector<std::ifstream> files;
    std::vector<std::string> filenames = { pathX, pathY, pathVelx, pathVely };

    for (const auto& filename : filenames) {
        // This line adds a new std::ifstream object to the files vector, opening the file specified by filename.
         //emplace_back(filename) constructs the std::ifstream object in place within the vector.
         //vector emplace_back() is a built-in method used to insert an element at the end of the vector
         //by constructing it in-place.
        files.emplace_back(filename);
        //back() just returns a reference to the last element
        if (!files.back().is_open()) {
            std::cerr << "Error:Cannot open " << filename << std::endl;
            break;
        }
        std::cout << "All files opend succesfully" << std::endl;
    }

    // std::ifstream file1(pathX);
    // std::ifstream file2(pathY);
    // std::ifstream file3(pathVelx);
    // std::ifstream file4(pathVely);

    // std::ifstream* files[] = { &file1,&file2,&file3,&file4 };
   //  const std::string filenames[] = { "file1","file2","file3","file4"};

    // for (int i = 0;i < 4;++i) {
     //    assert(files[i]->is_open() && ("Cannot open " + std::string(filenames[i])).c_str());
    // }

     //std::cout << "All files opend succesgfully" << std::endl;

    std::vector <double> vect;
    std::ofstream outputFile("C:/Users/lench/source/repos/NMC_Thesis/NMC_Thesis/output.txt");
    //The commented-out assert ensures that the network has no existing layers. 
    // If uncommented, the program would abort if layers were not empty.
    // TODO: What if it has layers already.
    //assert(layers.size() == 0 && "Cannot load to an already built nn.");
    layers.clear();

    std::string line;

    //Read file line by line
    for (int i = 0;i < 4;++i) {
        while (std::getline(files[i], line)) {
            std::istringstream iss(line);// Convert line to stream
            std::string value;

            //This reads each value 
            while (iss >> value) {
                try {
                    std::string output = cleanAndConvertToDouble(value);
                    if (!output.empty()) {
                        vect.push_back(std::stod(output));
                    }
                }
                catch (...) {
                }
            }
            // Write to output file with tab separation
       
            outputFile << "\n";
        }
        

        }
    for (int j = 0; j < vect.size(); ++j) {
        outputFile << vect[j];
        outputFile <<"                           " << vect[j] << "\n";
    }
    //Close file
    outputFile.close();
}


       // std::string temp;
       // double x;
        //iss >> temp >> x;

    //Reads some basic info
    //Reads the trained boolean (or some status flag) and data_index from the file.

  //  file.read((char*)(&trained), sizeof trained);
   // file.read((char*)(&data_index), sizeof data_index);

   // int layer_count;
    //Reads the number of layers stored in the file into layer_count.
   // file.read((char*)&layer_count, sizeof layer_count);
    //assert(layer_count >= 0);


    //Reading each layer
    //for (int i = 0; i < layer_count; i++) {
        //Creates an empty Layer object to store the data being loaded.
      //  Layer l;
        
        //Reads the bias matrix for this layer using read_matrix(file).
      //  l.biased = read_matrix(file);
      //  assert(l.biased.rows() == 1);
        //Initializing the Outputs Matrix
      //  l.outputs.init(1, l.biased.cols());
      //  Reads the weight matrix for this layer using read_matrix(file).
      //  l.weights = read_matrix(file);

        //Moves the newly constructed Layer into the layers vector.
       // layers.push_back(std::move(l));
   // }

    // Assert the dimentions are valid.
    //for (size_t i = 0; i < layers.size() - 1; i++) {
    //    const Layer& curr = layers[i];
    //    const Layer& next = layers[i + 1];
        //Ensures that the number of rows in the weight matrix matches the number of neurons in the current layer.
    //    assert(curr.weights.rows() == curr.outputs.cols());
        //Ensures that the number of columns in the weight matrix matches the number of neurons in the next layer.
     //   assert(curr.weights.cols() == next.outputs.cols());
    //}


//}
