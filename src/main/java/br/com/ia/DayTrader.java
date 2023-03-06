package br.com.ia;

import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.BackpropType;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.layers.*;
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

public class DayTrader {

    private static final int INPUT_SIZE = 1;
    private static final int HIDDEN_SIZE = 80;
    private static final int OUTPUT_SIZE = 1;
    private static final int DENSE_SIZE = 90;
    private static final int SEC_DENSE_SIZE = 70;
    private static final int THIRD_DENSE_SIZE = 50;
    private static final int FOUR_DENSE_SIZE = 45;

    public static double LEARNING_RATE = 1;
    private static final double L2_REGULARIZATION = 0.01;
    private double lastPrediction = 0;

    public int timePeriod = 180;
    private MultiLayerNetwork model;
    private Stock stock;
    private MultiLayerConfiguration config;
    private static INDArray dataSet;

    public DayTrader() throws IOException {
        config = new NeuralNetConfiguration.Builder()
                .weightInit(WeightInit.RELU)
                .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
                .updater(new Adam(LEARNING_RATE))
                .l2(L2_REGULARIZATION)
                .list()
                .layer(0, new LSTM.Builder().nIn(INPUT_SIZE).nOut(HIDDEN_SIZE).dropOut(0.2).build())
                .layer(1, new LSTM.Builder().nIn(HIDDEN_SIZE).nOut(HIDDEN_SIZE).dropOut(0.2).build())
                .layer(2, new DenseLayer.Builder().nIn(HIDDEN_SIZE).nOut(DENSE_SIZE)
                        .activation(Activation.LEAKYRELU).dropOut(0.5).build())
                .layer(3, new DenseLayer.Builder().nIn(DENSE_SIZE).nOut(SEC_DENSE_SIZE)
                        .activation(Activation.LEAKYRELU).dropOut(0.5).build())
                .layer(4, new DenseLayer.Builder().nIn(SEC_DENSE_SIZE).nOut(THIRD_DENSE_SIZE)
                        .activation(Activation.LEAKYRELU).dropOut(0.5).build())
                .layer(5, new RnnOutputLayer.Builder().nIn(THIRD_DENSE_SIZE).nOut(OUTPUT_SIZE)
                        .activation(Activation.TANH).lossFunction(LossFunctions.LossFunction.MSE).build())
                .backpropType(BackpropType.TruncatedBPTT)
                .tBPTTLength(300)
                .build();

        model = new MultiLayerNetwork(config);
        //model.getLayerWiseConfigurations().setValidateOutputLayerConfig(false);
        model.init();
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

    public void setDataSet(Stock stock) throws IOException {
        double[][] output = getOutputLabels(stock, timePeriod);

        int numOutputs = output.length * output[0].length;

        double[] outputFlat = new double[numOutputs];
        for (int i = 0; i < output.length; i++) {
            for (int j = 0; j < output[i].length; j++) {
                outputFlat[i * output[i].length + j] = output[i][j];
            }
        }

        dataSet = Nd4j.create(outputFlat, new int[]{1, output.length, output[0].length});
    }

    public INDArray getDataSet() {
        return dataSet;
    }

    public Stock getStock() {
        return stock;
    }

    public void setStock(Stock stock) {
        this.stock = stock;
    }

    public MultiLayerNetwork getModel() {
        return model;
    }

    public void setModel(MultiLayerNetwork model) {
        this.model = model;
    }

    public MultiLayerConfiguration getConfig() {
        return config;
    }

    public void setConfig(MultiLayerConfiguration config) {
        this.config = config;
    }

    public boolean hasPrediction() {
        boolean debounce;
        if (lastPrediction == 0) {
            debounce = false;
        } else {
            debounce = true;
        }

        return debounce;
    }

    public double getLastPrediction() {
        return lastPrediction;
    }
    public void setLastPrediction(double lastPrediction) {
        this.lastPrediction = lastPrediction;
    }
}
