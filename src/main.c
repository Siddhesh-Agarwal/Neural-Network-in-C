
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#define RAND_HIGH_RANGE (0.10)
#define RAND_MIN_RANGE (-0.10)
#define INIT_BIASES (0.0)

#define BATCH_SIZE 300
#define INPUT_LAYER_1_SIZE 2
#define INPUT_LAYER_2_SIZE 3
#define OUTPUT_LAYER_SIZE 3

typedef void (*activation_callback)(double *output);

typedef struct
{
    double *weights;
    double *biases;
    double *output;
    int input_size;
    int output_size;
    activation_callback callback;
} layer_dense_t;

typedef struct
{
    double *x;
    double *y;
} spiral_data_t;

double dot_product(double *input, double *weights, double *bias, int input_size, activation_callback callback)
{
    int i = 0;
    double output = 0.0;
    for (i = 0; i < input_size; i++)
    {
        output += input[i] * weights[i];
    }
    if (callback != NULL)
    {
        callback(&output);
    }
    output += *bias;
    return output;
}

void layer_output(double *input, double *weights, double *bias, int input_size, double *outputs, int output_size, activation_callback callback)
{
    int i = 0;
    int offset = 0;
    for (i = 0; i < output_size; i++)
    {
        outputs[i] = dot_product(input, weights + offset, &bias[i], input_size, callback);
        offset += input_size;
    }
}

void forward(layer_dense_t *previous_layer, layer_dense_t *next_layer)
{
    layer_output((previous_layer->output), next_layer->weights, next_layer->biases, next_layer->input_size, (next_layer->output), next_layer->output_size, next_layer->callback);
}
double rand_range(double min, double max)
{
    double range = (max - min);
    double div = RAND_MAX / range;
    return min + (rand() / div);
}

void layer_init(layer_dense_t *layer, int intput_size, int output_size)
{

    layer->input_size = intput_size;
    layer->output_size = output_size;

    // create data as a flat 1-D dataset
    layer->weights = malloc(sizeof(double) * intput_size * output_size);
    if (layer->weights == NULL)
    {
        printf("weights mem error\n");
        return;
    }
    layer->biases = malloc(sizeof(double) * output_size);
    if (layer->biases == NULL)
    {
        printf("biases mem error\n");
        return;
    }
    layer->output = malloc(sizeof(double) * output_size);

    if (layer->output == NULL)
    {
        printf("output mem error\n");
        return;
    }

    int i = 0;
    for (i = 0; i < (output_size); i++)
    {
        layer->biases[i] = INIT_BIASES;
    }
    for (i = 0; i < (intput_size * output_size); i++)
    {
        layer->weights[i] = rand_range(RAND_MIN_RANGE, RAND_HIGH_RANGE);
    }
}

void deloc_layer(layer_dense_t *layer)
{
    if (layer->weights != NULL)
    {
        free(layer->weights);
    }
    if (layer->biases != NULL)
    {
        free(layer->biases);
    }
    if (layer->biases != NULL)
    {
        free(layer->output);
    }
}

double activation_sigmoid(double x)
{
    double result;
    result = 1 / (1 + exp(-x));
    return result;
}

double activation_ReLU(double x)
{
    if (x < 0.0)
    {
        x = 0.0;
    }
    return x;
}

void activation1(double *output)
{
    *output = activation_ReLU(*output);
    *output = activation_sigmoid(*output);
}

double uniform_distribution(double rangeLow, double rangeHigh)
{
    double rng = rand() / (1.0 + RAND_MAX);
    double range = rangeHigh - rangeLow + 1;
    double rng_scaled = (rng * range) + rangeLow;
    return rng_scaled;
}

void spiral_data(int points, int classes, spiral_data_t *data)
{

    data->x = (double *)malloc(sizeof(double) * points * classes * 2);
    if (data->x == NULL)
    {
        printf("data mem error\n");
        return;
    }
    data->y = (double *)malloc(sizeof(double) * points * classes);
    if (data->y == NULL)
    {
        printf("points mem error\n");
        return;
    }
    int ix = 0;
    int iy = 0;
    int class_number = 0;
    for (class_number = 0; class_number < classes; class_number++)
    {
        double r = 0;
        double t = class_number * 4;

        while (r <= 1 && t <= (class_number + 1) * 4)
        {
            double random_t = t + uniform_distribution(-1.0, 1.0) * 0.2;

            data->x[ix] = r * sin(random_t * 2.5);
            data->x[ix + 1] = r * cos(random_t * 2.5);

            data->y[iy] = class_number;

            r += 1.0f / (points - 1);
            t += 4.0f / (points - 1);
            iy++;
            ix += 2;
        }
    }
}

void deloc_spiral(spiral_data_t *data)
{
    if (data->x != NULL)
    {
        free(data->x);
    }
    if (data->y != NULL)
    {
        free(data->y);
    }
}

void activation_softmax(layer_dense_t *output_layer)
{
    double sum = 0.0;
    double maxu = 0.0;
    int i = 0;

    maxu = output_layer->output[0];
    for (i = 1; i < output_layer->output_size; i++)
    {
        if (output_layer->output[i] > maxu)
        {
            maxu = output_layer->output[i];
        }
    }

    for (i = 0; i < output_layer->output_size; i++)
    {
        output_layer->output[i] = exp(output_layer->output[i] - maxu);
        sum += output_layer->output[i];
    }

    for (i = 0; i < output_layer->output_size; i++)
    {
        output_layer->output[i] = output_layer->output[i] / sum;
    }
}

double sum_softmax_layer_output(layer_dense_t *output_layer)
{
    double sum = 0.0;
    int i = 0;

    for (i = 0; i < output_layer->output_size; i++)
    {
        sum += output_layer->output[i];
    }

    return sum;
}

int main()
{
    srand(0);

    int i = 0;
    int j = 0;
    spiral_data_t X_data;
    layer_dense_t X;
    layer_dense_t dense1;
    layer_dense_t dense2;

    spiral_data(100, 3, &X_data);
    if (X_data.x == NULL)
    {
        printf("data null\n");
        return 0;
    }

    X.callback = NULL;
    dense1.callback = activation1;
    dense2.callback = NULL;

    layer_init(&dense1, INPUT_LAYER_1_SIZE, INPUT_LAYER_2_SIZE);
    layer_init(&dense2, INPUT_LAYER_2_SIZE, OUTPUT_LAYER_SIZE);

    for (i = 1; i <= BATCH_SIZE; i++)
    {
        X.output = &X_data.x[i * 2];
        forward(&X, &dense1);
        forward(&dense1, &dense2);

        activation_softmax(&dense2);

        printf("batch: %3d layer2_softmax: ", i);
        for (j = 0; j < dense2.output_size; j++)
        {
            printf("%lf ", dense2.output[j]);
        }
        printf("\n");
    }

    deloc_layer(&dense1);
    deloc_layer(&dense2);
    deloc_spiral(&X_data);
    return 0;
}
