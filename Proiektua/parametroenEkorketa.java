package Proiektua;
import weka.classifiers.Evaluation;
import weka.classifiers.bayes.BayesNet;
import weka.classifiers.bayes.net.search.SearchAlgorithm;
import weka.classifiers.bayes.net.search.local.K2;
import weka.classifiers.bayes.net.search.local.HillClimber;
import weka.classifiers.bayes.net.search.local.TabuSearch;
import weka.classifiers.bayes.net.search.local.TAN;
import weka.classifiers.bayes.net.estimate.SimpleEstimator;
import weka.classifiers.bayes.net.estimate.MultiNomialBMAEstimator;
import weka.classifiers.bayes.net.estimate.BMAEstimator;
import weka.classifiers.bayes.net.estimate.BayesNetEstimator;
import weka.core.Instances;
import weka.core.converters.ConverterUtils.DataSource;
import weka.filters.Filter;
import weka.filters.unsupervised.instance.Randomize;
import weka.filters.unsupervised.instance.RemovePercentage;
import java.io.FileWriter;
import java.io.IOException;



public class parametroenEkorketa {
    public static void main(String[] args) throws Exception {
        DataSource source = new DataSource(args[0]);
        Instances data = source.getDataSet();
        if (data.classIndex() == -1) {
            data.setClassIndex(data.numAttributes() - 1);
        }

        // Search algorithm list
        String[] searchAlgorithms = {"K2", "HillClimber", "TabuSearch"};

        double[] alphas = {0.1, 0.5, 1.0};
        int[] parents = {1,2,3,4,5};
        double bestFMeasure = -1;
        String estimatorName = null;
        String bestSAlgorithm = null;
        Double bestAlpha = null;
        String bestEstimator = null;
        int bestParent=0;
        int iteraciones = 0;

        for (String searchAlgorithm : searchAlgorithms) {
            for (int j=1;j<parents.length+1 ; j++) {
                for(int i=0; i<2; i++) {
                    for (double alpha : alphas) {
                        iteraciones++;
                        try {
                            BayesNetEstimator estimator = null;
                            if (i == 0){
                                estimator = new SimpleEstimator();
                                estimatorName = "SimpleEstimator";
                            } else if (i == 1) {
                                estimator = new MultiNomialBMAEstimator();
                                estimatorName = "MultiNomialBMAEstimator";
                            }

                            Randomize randomize = new Randomize();
                            randomize.setInputFormat(data);
                            Instances randomizedData = Filter.useFilter(data, randomize);
                            RemovePercentage removePercentage = new RemovePercentage();
                            removePercentage.setInputFormat(randomizedData);
                            removePercentage.setPercentage(70);  // 70% for training
                            Instances trainData = Filter.useFilter(randomizedData, removePercentage);
                            removePercentage.setInvertSelection(true);
                            Instances testData = Filter.useFilter(randomizedData, removePercentage);
                            BayesNet bayesNet = new BayesNet();
                            bayesNet.setSearchAlgorithm(createSearchAlgorithm(searchAlgorithm,j));
                            estimator.setAlpha(alpha);
                            bayesNet.setEstimator(estimator);
                            bayesNet.buildClassifier(trainData);
                            bayesNet.buildClassifier(trainData);

                            // Evaluate
                            Evaluation eval = new Evaluation(trainData);
                            eval.evaluateModel(bayesNet, testData);
                            int[] classCounts = trainData.attributeStats(trainData.classIndex()).nominalCounts;
                            if (classCounts == null || classCounts.length == 0) {
                                throw new Exception("No class values found");
                            }
                            int minority = 0;
                            int minCount = Integer.MAX_VALUE;
                            for (int n = 0; n < classCounts.length; n++) {
                                if (classCounts[n] < minCount) {
                                    minCount = classCounts[n];
                                    minority = n;
                                }
                            }
                            // Verify minority index is valid
                            if (minority < 0 || minority >= classCounts.length) {
                                minority = 0;  // fallback to first class
                            }
                            Double fMeasure = eval.fMeasure(minority);
                            System.out.println(fMeasure);
                            if (fMeasure > bestFMeasure) {
                                bestFMeasure = fMeasure;
                                bestSAlgorithm = searchAlgorithm;
                                bestAlpha = alpha;
                                bestEstimator = estimatorName;
                                bestParent = j;
                            }
                        } catch (Exception e) {}
                    }
                }
            }
        }
        System.out.println("Iteraciones: " + iteraciones);
        try (FileWriter writer = new FileWriter("parametroak.txt")) {
            writer.write(String.format("%s,%.1f,%d",
                    bestSAlgorithm,
                    bestAlpha,
                    bestParent));

            System.out.println("Parámetros guardados en parametroak.txt: " +
                    bestSAlgorithm + ", " + bestAlpha + ", " + bestParent);
        } catch (IOException e) {
            System.err.println("Error al guardar el archivo: " + e.getMessage());
        }
    }
    private static SearchAlgorithm createSearchAlgorithm(String algorithmType,int parents) {
        switch (algorithmType) {
            case "K2":
                K2 k2 = new K2();
                k2.setMaxNrOfParents(parents);
                k2.setInitAsNaiveBayes(true);
                return k2;

            case "HillClimber":
                HillClimber hc = new HillClimber();
                hc.setMaxNrOfParents(parents);
                hc.setInitAsNaiveBayes(true);
                return hc;

            case "TabuSearch":
                TabuSearch ts = new TabuSearch();
                ts.setMaxNrOfParents(parents);
                ts.setInitAsNaiveBayes(true);
                return ts;
            default:
                throw new IllegalArgumentException("Algoritmo no soportado: " + algorithmType);
        }
    }
}