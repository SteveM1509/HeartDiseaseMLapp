from matplotlib import pyplot
from math import cos, sin, atan, sqrt

def create_neuron(x, y):
    return {'x': x, 'y': y}

def draw_neuron(neuron, neuron_radius):
    circle = pyplot.Circle((neuron['x'], neuron['y']), radius=neuron_radius, fill=False,edgecolor='black')
    pyplot.gca().add_patch(circle)

def line_between_two_neurons(neuron1, neuron2, neuron_radius):
    angle = atan((neuron2['x'] - neuron1['x']) / float(neuron2['y'] - neuron1['y']))
    x_adjustment = neuron_radius * sin(angle)
    y_adjustment = neuron_radius * cos(angle)
    line = pyplot.Line2D((neuron1['x'] + x_adjustment, neuron2['x'] - x_adjustment),
                         (neuron1['y'] + y_adjustment, neuron2['y'] - y_adjustment))
    pyplot.gca().add_line(line)

def create_layer(network, number_of_neurons, number_of_neurons_in_widest_layer, prev_layer=None):
    vertical_distance_between_layers = 6
    horizontal_distance_between_neurons = 2
    neuron_radius = 0.5
    y = prev_layer['y'] - vertical_distance_between_layers if prev_layer else 0

    neurons = []
    x = horizontal_distance_between_neurons * (number_of_neurons_in_widest_layer - number_of_neurons) / 2
    for _ in range(number_of_neurons):
        neuron = create_neuron(x, y)
        neurons.append(neuron)
        x += horizontal_distance_between_neurons

    return {
        'vertical_distance_between_layers': vertical_distance_between_layers,
        'horizontal_distance_between_neurons': horizontal_distance_between_neurons,
        'neuron_radius': neuron_radius,
        'number_of_neurons_in_widest_layer': number_of_neurons_in_widest_layer,
        'previous_layer': prev_layer,
        'y': y,
        'neurons': neurons
    }

def draw_layer(layer, layer_type=0):
    if layer['previous_layer']:
        for neuron in layer['neurons']:
            for prev_neuron in layer['previous_layer']['neurons']:
                line_between_two_neurons(neuron, prev_neuron, layer['neuron_radius'])

    for neuron in layer['neurons']:
        draw_neuron(neuron, layer['neuron_radius'])

    x_text = layer['number_of_neurons_in_widest_layer'] * layer['horizontal_distance_between_neurons']
    y_text = layer['y'] - layer['vertical_distance_between_layers'] * 0.1
    if layer_type == 0:
        pyplot.text(x_text, y_text, 'Input Layer', fontsize=12)
    elif layer_type == -1:
        pyplot.text(x_text, y_text, 'Output Layer', fontsize=12)
    else:
        pyplot.text(x_text, y_text, f'Hidden Layer {layer_type}', fontsize=12)


def create_network(neurons_list):
    widest_layer = max(neurons_list)
    layers = []
    prev_layer = None
    for i, neurons in enumerate(neurons_list):
        layer = create_layer(layers, neurons, widest_layer, prev_layer)
        layers.append(layer)
        prev_layer = layer
    return layers

def draw_network(layers):
    widest_layer_neurons = max(layer['number_of_neurons_in_widest_layer'] for layer in layers)
    fig_width = widest_layer_neurons * 2 
    fig_height = len(layers) * 2 

    fig = pyplot.figure(figsize=(fig_width, fig_height))
    for i, layer in enumerate(layers):
        draw_layer(layer, i if i != len(layers) - 1 else -1)

    pyplot.axis('scaled')
    pyplot.axis('off')
    pyplot.title('Neural Network architecture', fontsize=15)
    return fig