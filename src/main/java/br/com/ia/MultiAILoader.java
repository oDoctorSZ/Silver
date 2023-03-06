package br.com.ia;

import br.com.ia.methods.TrainingManager;
import br.com.ia.utils.DataSplitter;
import br.com.ia.utils.Utils;
import org.deeplearning4j.datasets.iterator.utilty.ListDataSetIterator;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.factory.Nd4j;
import yahoofinance.Stock;
import yahoofinance.histquotes.HistoricalQuote;
import yahoofinance.histquotes.Interval;

import javax.xml.crypto.Data;
import java.io.IOException;
import java.util.*;
import java.util.concurrent.atomic.AtomicReference;
import java.util.stream.Collectors;

public class MultiAILoader {

    public static INDArray predicted;
    public static INDArray combinedPrediction;
    public static double predictedPrice;
    public static int modelsSize = 0;
    public static int epochs = 250;
    public static int epochs2 = 100000;
    private static INDArray val;
    private static ArrayList<DayTrader> modelsFinal = new ArrayList<>();
    private static HashMap<DayTrader, Double> predictedPrices = new HashMap<>();

    public static void Judge(List<DayTrader> models, DataSet dataSet, Stock stock) throws IOException {

        double[] currentPrice = new double[1];

        int sequenceLength = 15;
        int trainPercentage = 80;

        for (DayTrader model : models) {
            if (model.hasPrediction()) {
                currentPrice[0] = model.getLastPrediction();
            } else {
                currentPrice[0] = model.getStock().getQuote().getPrice().doubleValue();
            }

            model.getModel().fit(dataSet);
        }

        Calendar from = Calendar.getInstance();
        from.add(Calendar.YEAR, -1);
        Calendar to = Calendar.getInstance();
        List<HistoricalQuote> quoteData = stock.getHistory(from, to, Interval.DAILY);

        Map<String, List<HistoricalQuote>> quotesByDate = quoteData.stream()
                .collect(Collectors.groupingBy(q -> q.getDate().toString()));

        Map<String, List<HistoricalQuote>> quotesBy15Min = new HashMap<>();

        List<HistoricalQuote> quotes15Min = new ArrayList<>();

        quotesByDate.forEach((date, dailyQuotes) -> {

            for (int i = 0; i < dailyQuotes.size(); i += 4) {
                HistoricalQuote quote = dailyQuotes.get(i);
                quotes15Min.add(quote);
            }

            quotesBy15Min.put(date, quotes15Min);
        });

        val = Nd4j.create(new double[] { currentPrice[0] }, new int[] {1, 1, 1});

        for (int i = 0; i < epochs2; i++) {

            Random random = new Random();
            int rand = random.nextInt(0, 200);
            //ListDataSetIterator iterator = new ListDataSetIterator(Collections.singletonList(splitTrainingDataSet));

            DataSet splitTrainingDataSet15 = (DataSplitter.prepareData15Min(quotes15Min, sequenceLength, trainPercentage, 0.5));
            DataSet splitTrainingDataSet = (DataSplitter.prepareData(stock, quoteData, sequenceLength, trainPercentage, 0.5));
            List<DataSet> trainingData = Arrays.asList(splitTrainingDataSet);
            DataSetIterator trainIterator = new ListDataSetIterator<>(trainingData, 10);

            for (DayTrader model : models) {

                model.getModel().setListeners(new ScoreIterationListener(1));
                //model.getModel().fit(splitTrainingDataSet15);
                model.getModel().fit(splitTrainingDataSet);
                model.getModel().fit(dataSet);

                Utils.dataSetRef.put(model.getStock(), splitTrainingDataSet);

            }

        }

        models.forEach(model -> {
            try {
                model.getModel().feedForward(val);
                predictedPrice = model.getModel().output(model.getDataSet()).getDouble(0);
                predictedPrices.put(model, predictedPrice);

            } catch (Exception e) {
                throw new RuntimeException(e);
            }
        });

        while (modelsFinal.size() < 5) {
            generateVal(models, currentPrice[0], epochs, dataSet);
        }

        predicted = Nd4j.zeros(modelsFinal.size(), 1);

        for (int j = 0; j < modelsFinal.size(); j++) {
            DayTrader model = modelsFinal.get(j);
            predicted.putScalar(j, model.getModel().output(model.getDataSet()).getDouble(0));

            model.getModel().clearLayersStates();
        }

        combinedPrediction = predicted.mean(0);
        predictedPrice = combinedPrediction.getDouble(0);

        models.forEach(model -> {
            model.setLastPrediction(predictedPrice);
        });

        if (predictedPrice > currentPrice[0]) {
            System.out.println("Comprar em " + models.get(0).getStock() + " Valor: " + predictedPrice + "!");
        } else {
            System.out.println("Vender em " + models.get(0).getStock() + " Valor: " + predictedPrice + "!");
        }

        modelsFinal.clear();
        predictedPrice = 0;
        combinedPrediction.close();

    }


    private static void generateVal(List<DayTrader> models, double currentPrice, int times, DataSet dataSet) {
        for (int i = 0; i < times; i++) {

            models.forEach(model -> {
                predictedPrice = predictedPrices.get(model);

                if (predictedPrice > (currentPrice + 0.004) || predictedPrice < (currentPrice - 0.004)) {

                    val = Nd4j.create(new double[] { currentPrice }, new int[] {1, 1, 1});

                    //predictedPrice = model.getModel().output(model.getDataSet()).getDouble(0);
                    predictedPrice = model.getModel().rnnTimeStep(val).getDouble(0);
                    predictedPrices.replace(model, predictedPrice);

                    model.getModel().fit(dataSet);
                    model.getModel().updateRnnStateWithTBPTTState();

                    System.out.println(predictedPrice);

                } else {
                    modelsFinal.add(model);

                }

                modelsSize++;

            });

        }
    }




}
