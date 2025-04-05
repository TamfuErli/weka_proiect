package Proiektua;

import weka.classifiers.Classifier;
import weka.classifiers.Evaluation;
import weka.classifiers.bayes.NaiveBayes;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.SerializationHelper;
import weka.core.converters.ConverterUtils.DataSource;
import weka.filters.Filter;
import weka.filters.supervised.instance.Resample;

import java.io.FileWriter;
import java.util.Random;

public class Baseline {

    public static void main(String args[]) {
        try {
            String trainBowFss = args[0];
            String hiztegi = args[1];
            String devCsv= args[2];
            String modeloa = args[3];
            String ebaluazioa = args[4];
            String predictionsPath = args[5];
            String devCsvOna = args[6];
            String csvArff = args[7];
            String devBowFss = args[8];

            DataSource source = new DataSource(trainBowFss);
            Instances train_bow_fss = source.getDataSet();
            train_bow_fss.setClassIndex(train_bow_fss.numAttributes()-1);

            //klase minoritarioa ateratzeko
            System.out.println(train_bow_fss.attributeStats(train_bow_fss.classIndex()).nominalCounts);
            int[] classCounts = train_bow_fss.attributeStats(train_bow_fss.classIndex()).nominalCounts;
            if (classCounts == null || classCounts.length == 0) {
                throw new Exception("No class values found");
            }
            int minClassIndex = 0;
            int minCount = Integer.MAX_VALUE;
            for (int n = 0; n < classCounts.length; n++) {
                if (classCounts[n] < minCount) {
                    minCount = classCounts[n];
                    minClassIndex = n;
                }
            }
            Instances devRaw = csvCleanToArff.bihurketaArff(devCsv, devCsvOna, csvArff);
            Instances dev_bow_fss = FssDevTest.applyFssWithDictionary(devRaw, hiztegi);
            Fss.saveArff(dev_bow_fss, devBowFss);
//

            NaiveBayes nb = new NaiveBayes();
            nb.buildClassifier(train_bow_fss);
            SerializationHelper.write(modeloa, nb);

            try (FileWriter fw = new FileWriter(ebaluazioa)){
                fw.write(" -----------EZ_ZINTZOA---------------\n");
                Evaluation eval = new Evaluation(train_bow_fss);
                eval.evaluateModel(nb,train_bow_fss);
                fw.write(eval.toMatrixString() + "\n");
                fw.write(("Klasse minoritarioaren f-measure: " + eval.fMeasure(minClassIndex)));
                fw.write("\n");
                fw.write(("Klasse minoritarioaren precision: " + eval.precision(minClassIndex)));

                fw.write("\n");

                fw.write(" -----------10_FOLD_CROSS-VALIDATION---------------");
                eval.crossValidateModel(new NaiveBayes(), train_bow_fss, 10, new Random(1));
                fw.write(eval.toMatrixString() + "\n");
                fw.write(("Klasse minoritarioaren f-measure: " + eval.fMeasure(minClassIndex)));
                fw.write("\n");

                fw.write(("Klasse minoritarioaren precision: " + eval.precision(minClassIndex)));

                fw.write("\n");


                fw.write(" -----------REPEATED_STRATIFIED_HOLD_OUT---------------\n");
                repeatedStratifiedHoldOut(train_bow_fss, new NaiveBayes(), minClassIndex, ebaluazioa);
                NaiveBayes model = (NaiveBayes) SerializationHelper.read(modeloa);
                predikzioakEgin(predictionsPath, model, dev_bow_fss);
            }
        } catch (Exception e) {
            System.out.println("ERROREA prozesuan");
            e.printStackTrace();
        }
    }

    public static void repeatedStratifiedHoldOut(Instances dataTrain, Classifier model, int minClassIndex, String predictionPath) throws Exception {
        int rep = 5;
        double[][] totalConfusionMatrix = new double[dataTrain.classAttribute().numValues()]
                [dataTrain.classAttribute().numValues()];
        double totalFMeasure = 0.0;
        double totalPrecision = 0.0;
        double totalRecall = 0.0;
        double ehuneko = 70.0;

        try (FileWriter fw = new FileWriter(predictionPath, true)) {
            fw.write("\n----------- REPEATED STRATIFIED HOLD OUT ("+rep+" repeticiones) -----------\n");

            for (int i = 1; i <= rep; i++) {
                Instances datuak = new Instances(dataTrain); // Copia para cada iteración
                datuak.randomize(new Random(i));

                Resample filtroa = new Resample();
                filtroa.setNoReplacement(true);
                filtroa.setSampleSizePercent(ehuneko);
                filtroa.setInputFormat(datuak);
                Instances trainData = Filter.useFilter(datuak, filtroa);

                filtroa.setInvertSelection(true);
                Instances devData = Filter.useFilter(datuak, filtroa);

                model.buildClassifier(trainData); // Entrenar con trainData, no con datuak
                Evaluation evaluation = new Evaluation(trainData);
                evaluation.evaluateModel(model, devData);

                // Acumular métricas
                totalConfusionMatrix = matrizeakGehitu(totalConfusionMatrix, evaluation.confusionMatrix());
                totalFMeasure += evaluation.fMeasure(minClassIndex);
                totalPrecision += evaluation.precision(minClassIndex);
                totalRecall += evaluation.recall(minClassIndex);

                // Escribir resultados de cada iteración
                fw.write("\n--- Iteración " + i + " ---\n");
                fw.write("Matriz de Confusión:\n" + evaluation.toMatrixString() + "\n");
                fw.write(String.format("Clase minoritaria - Precisión: %.4f, Recall: %.4f, F-measure: %.4f\n",
                        evaluation.precision(minClassIndex),
                        evaluation.recall(minClassIndex),
                        evaluation.fMeasure(minClassIndex)));
            }

            // Calcular promedios
            double[][] avgConfusionMatrix = matrizDividida(totalConfusionMatrix, rep);
            double avgFMeasure = totalFMeasure / rep;
            double avgPrecision = totalPrecision / rep;
            double avgRecall = totalRecall / rep;

            // Escribir resultados agregados
            fw.write("\n=== RESULTADOS PROMEDIO ===\n");
            fw.write("Matriz de Confusión Promedio:\n");
            for (int i = 0; i < avgConfusionMatrix.length; i++) {
                for (int j = 0; j < avgConfusionMatrix[i].length; j++) {
                    fw.write(String.format("%.1f\t", avgConfusionMatrix[i][j]));
                }
                fw.write("\n");
            }

            fw.write(String.format("\nClase minoritaria (índice %d):\n", minClassIndex));
            fw.write(String.format("Precisión promedio: %.4f\n", avgPrecision));
            fw.write(String.format("Recall promedio: %.4f\n", avgRecall));
            fw.write(String.format("F-measure promedio: %.4f\n", avgFMeasure));
        }
    }

    public static double[][] matrizeakGehitu(double[][] totalMatrix, double[][] nMatrix){
        double[][] resultMatrix = new double[totalMatrix.length][totalMatrix[0].length];
        for(int i=0; i< totalMatrix.length; i++){
            for (int j=0; j<totalMatrix[i].length; j++){
                resultMatrix[i][j] = totalMatrix[i][j] + nMatrix[i][j];
            }
        }
        return resultMatrix;
    }

    public static double[][] matrizDividida(double [][] totalMatrix, int rep){
        double[][] resultMatrix = new double[totalMatrix.length][totalMatrix[0].length];
        for(int i=0; i< totalMatrix.length; i++){
            for (int j=0; j<totalMatrix[i].length; j++){
                resultMatrix[i][j] = totalMatrix[i][j] / rep;
            }
        }
        return resultMatrix;
    }

    public static void predikzioakEgin(String predictionPath, Classifier model,
                                       Instances test_bow_fss) throws Exception {
        try(FileWriter fw = new FileWriter(predictionPath)) {

            for (Instance inst : test_bow_fss) {
                int i = (int) model.classifyInstance(inst);
                inst.setClassValue(test_bow_fss.classAttribute().value(i));
                fw.write(inst + "\n");
            }
        }
    }
}














