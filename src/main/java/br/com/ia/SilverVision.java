package br.com.ia;

import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.BackpropType;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.Updater;
import org.deeplearning4j.nn.conf.layers.LSTM;
import org.deeplearning4j.nn.conf.layers.LossLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.conf.layers.RnnOutputLayer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.learning.config.Adam;
import org.nd4j.linalg.lossfunctions.LossFunctions;
import yahoofinance.Stock;
import yahoofinance.histquotes.HistoricalQuote;
import yahoofinance.histquotes.Interval;

import java.io.IOException;
import java.util.*;
import java.util.stream.Collectors;

public class SilverVision {

    int inputSize = 1;
    int hiddenSize = 20;
    int outputSize = 1;
    int nEpochs = 10;
    int nBatchSize = 32;
    double learningRate = 0.01;

    public int timePeriod = 180;
    private MultiLayerNetwork model;
    private Stock stock;
    private MultiLayerConfiguration config;

    public SilverVision() throws IOException {
        config = new NeuralNetConfiguration.Builder()
                .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
                .updater(new Adam(learningRate))
                .weightInit(WeightInit.XAVIER)
                .list()
                .layer(0, new LSTM.Builder()
                        .nIn(inputSize)
                        .nOut(hiddenSize)
                        .activation(Activation.TANH)
                        .build())
                .layer(1, new RnnOutputLayer.Builder()
                        .nIn(hiddenSize)
                        .nOut(outputSize)
                        .lossFunction(LossFunctions.LossFunction.MSE)
                        .activation(Activation.IDENTITY)
                        .build())
                .layer(2, new OutputLayer.Builder(LossFunctions.LossFunction.MSE)
                        .nIn(hiddenSize)
                        .nOut(outputSize)
                        .activation(Activation.IDENTITY)
                        .build())
                .backpropType(BackpropType.TruncatedBPTT)
                .tBPTTForwardLength(360)
                .tBPTTBackwardLength(360)
                .build();

        model = new MultiLayerNetwork(config);
        model.init();
    }
    public double[][] getTrainingData(Stock stock, int timePeriod) throws IOException {
        // Recupere os preços de fechamento das ações nos últimos "timePeriod" dias

        Calendar from = Calendar.getInstance();
        from.add(Calendar.YEAR, -1);

        Calendar to = Calendar.getInstance();

        List<HistoricalQuote> quotes = stock.getHistory(from, to, Interval.DAILY);

        Map<String, List<HistoricalQuote>> quotesByDate = quotes.stream()
                .collect(Collectors.groupingBy(q -> q.getDate().toString()));

        Map<String, List<HistoricalQuote>> quotesBy15Min = new HashMap<>();

        quotesByDate.forEach((date, dailyQuotes) -> {
            List<HistoricalQuote> quotes15Min = new ArrayList<>();

            for (int i = 0; i < dailyQuotes.size(); i += 4) {
                HistoricalQuote quote = dailyQuotes.get(i);
                quotes15Min.add(quote);
            }

            quotesBy15Min.put(date, quotes15Min);
        });

        int numQuotes = quotesBy15Min.values().size();
        double[][] trainingData = new double[1][numQuotes];
        for (int i = 0; i < numQuotes; i++) {
            trainingData[0][i] = quotes.get(i).getClose().doubleValue();
        }
        return trainingData;
    }

    private double[][] getOutputLabels(Stock stock, int timePeriod) throws IOException {
        // Recupere os preços de fechamento das ações nos próximos "timePeriod" dias
        Calendar from = Calendar.getInstance();
        from.add(Calendar.YEAR, -1);

        Calendar to = Calendar.getInstance();

        List<HistoricalQuote> quotes = stock.getHistory(from, to, Interval.DAILY);

        Map<String, List<HistoricalQuote>> quotesByDate = quotes.stream()
                .collect(Collectors.groupingBy(q -> q.getDate().toString()));

        Map<String, List<HistoricalQuote>> quotesBy15Min = new HashMap<>();

        quotesByDate.forEach((date, dailyQuotes) -> {
            List<HistoricalQuote> quotes15Min = new ArrayList<>();

            for (int i = 0; i < dailyQuotes.size(); i += 4) {
                HistoricalQuote quote = dailyQuotes.get(i);
                quotes15Min.add(quote);
            }

            quotesBy15Min.put(date, quotes15Min);
        });

        int numQuotes = quotesBy15Min.values().size();
        double[][] outputLabels = new double[1][numQuotes];
        for (int i = 0; i < numQuotes; i++) {
            outputLabels[0][i] = quotes.get(i).getClose().doubleValue();
        }
        return outputLabels;
    }
    public INDArray getDataSet() throws IOException {
        double[][] input = getTrainingData(stock, timePeriod);
        double[][] output = getOutputLabels(stock, timePeriod);

        int numInputs = input.length * input[0].length;
        int numOutputs = output.length * output[0].length;

        double[] inputFlat = new double[numInputs];
        for (int i = 0; i < input.length; i++) {
            for (int j = 0; j < input[i].length; j++) {
                inputFlat[i * input[i].length + j] = input[i][j];
            }
        }

        double[] outputFlat = new double[numOutputs];
        for (int i = 0; i < output.length; i++) {
            for (int j = 0; j < output[i].length; j++) {
                outputFlat[i * output[i].length + j] = output[i][j];
            }
        }

        INDArray inputND = Nd4j.create(inputFlat, new int[]{1, input.length, input[0].length});
        INDArray outputND = Nd4j.create(outputFlat, new int[]{1, output.length, output[0].length});

        return outputND;
    }
    public static double reward(double actual, double prediction, double learningRate) {
        if (prediction > (actual + 0.004)) {
            learningRate -= 0.001;
        } else if (prediction < (actual - 0.004)) {
            learningRate += 0.001;
        }
        return learningRate;
    }

    public int getInputSize() {
        return inputSize;
    }

    public void setInputSize(int inputSize) {
        this.inputSize = inputSize;
    }

    public int getHiddenSize() {
        return hiddenSize;
    }

    public void setHiddenSize(int hiddenSize) {
        this.hiddenSize = hiddenSize;
    }

    public int getOutputSize() {
        return outputSize;
    }

    public void setOutputSize(int outputSize) {
        this.outputSize = outputSize;
    }

    public MultiLayerNetwork getModel() {
        return model;
    }

    public void setModel(MultiLayerNetwork model) {
        this.model = model;
    }

    public Stock getStock() {
        return stock;
    }

    public void setStock(Stock stock) {
        this.stock = stock;
    }

    public MultiLayerConfiguration getConfig() {
        return config;
    }

    public void setConfig(MultiLayerConfiguration config) {
        this.config = config;
    }
}
