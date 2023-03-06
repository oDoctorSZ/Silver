package br.com.ia.methods;

import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.factory.Nd4j;
import yahoofinance.Stock;
import yahoofinance.histquotes.HistoricalQuote;
import yahoofinance.histquotes.Interval;

import java.io.IOException;
import java.util.*;
import java.util.stream.Collectors;

public class TrainingManager {


    public static HashMap<String, DataSet> dataSetMap = new HashMap<>();
    public static List<DataSet> dataSets = new ArrayList<>();
    public static Stack<List<HistoricalQuote>> quoteDataStack = new Stack<>();

    public static void loadDataSet(Stock stock) throws Exception {
        double[][] input = getTrainingData(stock);
        double[][] output = getOutputLabels(stock);

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
        DataSet dataSet = new DataSet(inputND, outputND);

        dataSetMap.put(stock.getName(), dataSet);
        dataSets.add(dataSet);
    }

    public static void trainWithDataSet(MultiLayerNetwork model) {

        dataSetMap.forEach((stockName, dataSet) -> {
            model.fit(dataSet);


        });

        System.out.println("Silver Instance has been trained with all the market actions!");

    }

    public static DataSet currDataSet(Stock stock) throws IOException {
        double[][] input = getTrainingData(stock);
        double[][] output = getOutputLabels(stock);

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

        DataSet dataSet = new DataSet(inputND, outputND);

        return dataSet;
    }

    public static double[][] getTrainingData(Stock stock) throws IOException {

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

        System.out.println(trainingData);

        return trainingData;
    }

    public static double[][] getOutputLabels(Stock stock) throws IOException {
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



}
