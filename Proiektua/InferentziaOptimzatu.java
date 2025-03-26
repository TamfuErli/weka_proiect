package Proiektua;

import weka.classifiers.Classifier;
import weka.classifiers.CostMatrix;
import weka.classifiers.Evaluation;
import weka.classifiers.bayes.BayesNet;
import weka.classifiers.bayes.net.search.SearchAlgorithm;
import weka.classifiers.bayes.net.search.global.K2;
import weka.classifiers.bayes.net.search.global.TAN;
import weka.classifiers.meta.CostSensitiveClassifier;
import weka.core.Instances;

public class InferentziaOptimzatu {
    /**
     * Klase maioritarioaren indizea eskurantzen du
     */
    public static int klaseMayoritarioa(Instances data) {
        int[] counts = data.attributeStats(data.classIndex()).nominalCounts;
        int maxIndex = 0;
        for (int i = 1; i < counts.length; i++) {
            if (counts[i] > counts[maxIndex]) maxIndex = i;
        }
        return maxIndex;
    }

    /**
     * Klase minoritarioaren indizea eskurantzen du
     */
    public static int klaseMinoritarioa(Instances data) {
        int[] counts = data.attributeStats(data.classIndex()).nominalCounts;
        int minIndex = 0;
        for (int i = 1; i < counts.length; i++) {
            if (counts[i] < counts[minIndex] && counts[i] > 0) minIndex = i;
        }
        return minIndex;
    }

    private static CostMatrix crearMatrizCostos(int numClasses, int claseMinoritaria) {
        CostMatrix matrix = new CostMatrix(numClasses);
        matrix.initialize();
        for (int i = 0; i < numClasses; i++) {
            for (int j = 0; j < numClasses; j++) {
                if (i != j && i == claseMinoritaria) {
                    matrix.setCell(i, j, 2.0); // Honekin clase minoritarioaren errorea x2 kostuarekin penalizatzen  da
                } else {
                    matrix.setCell(i, j, 1.0); // Error estándar
                }
            }
        }
        return matrix;
    }

    public static Classifier parametroakOptimizatu(Instances train, Instances dev) throws Exception{

        BayesNet modeloHoberena = new BayesNet();
        SearchAlgorithm bestSA = null;
        double bestFmeasure = -1;

        int minClassIndex = klaseMinoritarioa(dev);
        CostMatrix costMatrix = crearMatrizCostos(train.numClasses(), minClassIndex);

        SearchAlgorithm[] searchAlgorithms = {
                new K2(), new TAN()
        };

        for(SearchAlgorithm algorithm : searchAlgorithms) {
            BayesNet model = new BayesNet();
            model.setSearchAlgorithm(algorithm);

            CostSensitiveClassifier costModel = new CostSensitiveClassifier();
            costModel.setClassifier(model);
            costModel.setCostMatrix(costMatrix);
            costModel.buildClassifier(train);

            Evaluation eval = new Evaluation(dev, costMatrix);
            eval.evaluateModel(costModel, dev);
            double currentFMeasure = eval.fMeasure(minClassIndex);

            if (currentFMeasure > bestFmeasure) {
                bestFmeasure = currentFMeasure;
                bestSA = algorithm;
                modeloHoberena = model;
            }
        }
    }



}
