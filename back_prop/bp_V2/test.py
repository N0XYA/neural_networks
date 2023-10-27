import pandas as pd
import numpy as np
import random


def data_to_lists(df):
    labels_list = []
    pixel_vals_list = []
    for row in df:
        labels_list.append(row[0])
        row.pop(0)
        pixel_vals_list.append(row)
    return labels_list, pixel_vals_list


class nn:
    def __init__(self, list_of_labels, list_of_inputs, hid_size=80):
        self.labels = list_of_labels
        self.inputs = list_of_inputs
        self.hidden_layer_size = hid_size

    def generate_start_weights(self):
        input_weights = [0 for _ in range(self.hidden_layer_size)]
        bias = [1 for _ in range(self.hidden_layer_size)]
        hidden_weights = [0 for _ in range(10)]
        for i in range(self.hidden_layer_size):
            input_weights[i] = [random.random() for _ in range(len(self.inputs[0]))]
        for i in range(10):
            hidden_weights[i] = [random.random() for _ in range(self.hidden_layer_size)]
        np.save("inp_w", input_weights)
        np.save("hid_w", hidden_weights)
        np.save("bias", bias)

    def softmax(self, layer):
        """Compute softmax values for each sets of scores in x."""
        e_x = np.exp(layer - np.max(layer))
        return e_x / e_x.sum(axis=0)  # only difference

    def train(self):
        num_of_epoches = 10
        step = 0.001
        input_weights = np.load("inp_w.npy")
        hidden_weights = np.load("hid_w.npy")
        bias = np.load("bias.npy")
        # self.generate_start_weights()
        print("weights loaded!")


        for epoch in range(num_of_epoches):
            print("epoch:", epoch)
            # if epoch % 5 == 0:
            #     print("epoch:", epoch)

            for image_num in range(len(self.inputs)):
                inp_layer = self.inputs[image_num]
                label = self.labels[image_num]
                expected_out = [1 if label == _ else 0 for _ in range(10)]

                hidden_layer = []
                # inputs to hidden layer
                for neuron_num in range(self.hidden_layer_size):
                    hidden_out = []
                    for input_num in range(len(inp_layer)):
                        hidden_out.append((inp_layer[input_num] * input_weights[neuron_num][input_num]) + 1)
                    hidden_out = self.softmax(hidden_out)
                    hidden_layer.append(max(hidden_out))

                # hidden layer to outputs
                output = []
                for neuron_num in range(10):
                    out = []
                    for input_num in range(self.hidden_layer_size):
                        out.append((hidden_layer[input_num]*hidden_weights[neuron_num][input_num]) + 1)
                    out = self.softmax(out)
                    output.append(max(out))


                # error
                # sigmas
                errors = []
                out_sigmas = []
                omega = [0 for _ in range(self.hidden_layer_size)]
                for i in range(10):
                    error = (expected_out[i] - output[i])**2 / 2
                    sigma = error * output[i] * (1 - output[i])
                    out_sigmas.append(sigma)
                    for weight_num in range(len(hidden_weights[i])):
                        # updating hidden layer weights
                        #MAYBE change raws in place
                        hidden_weights[i][weight_num] -= step * sigma * hidden_layer[weight_num]
                        omega[weight_num] += sigma * hidden_weights[i][weight_num]

                for i in range(self.hidden_layer_size):
                    for weight_num in range(len(input_weights[i])):
                        input_weights[i][weight_num] -= step * omega[i] * inp_layer[weight_num]

            np.save("inp_w", input_weights)
            np.save("hid_w", hidden_weights)
            print("weights saved!")


def main():
    df = pd.read_csv("../5k.csv")
    df = df.values.tolist()
    labels_list, pixel_vals_list = data_to_lists(df)
    neural_network = nn(labels_list, pixel_vals_list)
    neural_network.train()
    return


if __name__ == "__main__":
    main()