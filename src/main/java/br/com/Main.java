package br.com;

import br.com.ia.DayTrader;
import br.com.ia.MultiAILoader;
import br.com.ia.utils.Utils;
import jdk.jshell.execution.Util;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;

import java.io.IOException;
import java.util.ArrayList;
import java.util.List;
import java.util.Scanner;

public class Main {

    public static String stockSymbol = "";

    private static void initializer() throws Exception {
       //Utils.initializeMarketStock();
       //Utils.initializerSilverTrainer();
    }

    public static void main(String[] args) throws Exception {

        initializer();

        System.out.println("Silver Vision has been initialized!");

        while (!(stockSymbol.equalsIgnoreCase("cancel"))) {

            Scanner board = new Scanner(System.in);

            System.out.println("Qual moeda você quer prever as ações?");

            stockSymbol = board.nextLine();

            System.out.println("Claro!");
            Utils.silverJudgeWithMultiAI(stockSymbol);


            board.reset();
        }

    }
}