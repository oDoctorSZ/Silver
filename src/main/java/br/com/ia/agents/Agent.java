package br.com.ia.agents;

import org.deeplearning4j.nn.conf.NeuralNetConfiguration;

import java.util.Random;

public class Agent {

    private static final int INPUT_SIZE = 100; // tamanho da notícia
    private static final int OUTPUT_SIZE = 3; // número de ações possíveis
    private static final int HIDDEN_SIZE = 64; // número de neurônios nas camadas ocultas
    private static final double LEARNING_RATE = 0.01; // taxa de aprendizagem
    private static final double DISCOUNT_FACTOR = 0.99; // fator de desconto
    private static final int BUFFER_SIZE = 10000; // tamanho do buffer de replay
    private static final int BATCH_SIZE = 64; // tamanho do batch de amostras


    private static final String SAVE_PATH = "models/dqn_news_agent.zip"; // caminho para salvar o modelo

    private static final Random random = new Random(); // gerador de números aleatórios




}
