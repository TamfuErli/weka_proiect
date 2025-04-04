package Proiektua;
import weka.core.*;
import weka.filters.Filter;
import weka.filters.unsupervised.attribute.Remove;
import weka.core.converters.ArffSaver;

import weka.attributeSelection.InfoGainAttributeEval;
import weka.attributeSelection.Ranker;
import weka.core.Attribute;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.converters.ArffSaver;
import weka.core.converters.CSVLoader;
import weka.filters.Filter;
import weka.filters.supervised.attribute.AttributeSelection;
import weka.filters.supervised.instance.Resample;
import weka.filters.unsupervised.attribute.*;
import java.io.File;
import java.io.FileReader;
import java.io.BufferedReader;
import java.io.FileWriter;
import java.util.ArrayList;
import java.util.List;

public class Fss {

    public static void main(String[] args) {
        try {
            // Cargar archivo ARFF
            String inputArff =args[0];  // Cambia la ruta según sea necesario
            String outputPath = args[1];
            String hiztegiBerria = args[2];

            BufferedReader reader = new BufferedReader(new FileReader(inputArff));

            Instances data = new Instances(reader);
            reader.close();

            // Establecer el índice de la clase (si es necesario)
            AttributeSelection filter = createAttributeFilter(2000); // 2000 mejores atributos

            // Aplicar filtro
            filter.setInputFormat(data);
            Instances filteredData = filter.useFilter(data, filter);

            // Guardar resultados
            saveArff(filteredData, outputPath);
            saveDictionary(filteredData, hiztegiBerria);

            System.out.println("\nProceso completado:");
            System.out.println("- ARFF filtrado guardado en: " + outputPath);
            System.out.println("- Diccionario óptimo guardado en: " + hiztegiBerria);
            System.out.println("\nDatos filtrados:");
            System.out.println(filteredData.toSummaryString());

        } catch (Exception e) {
            System.err.println("Error durante el procesamiento:");
            e.printStackTrace();
            System.exit(1);
        }
    }

    private static AttributeSelection createAttributeFilter(int numAttributes) throws Exception {
        InfoGainAttributeEval evaluator = new InfoGainAttributeEval();
        Ranker ranker = new Ranker();
        ranker.setNumToSelect(numAttributes);
        ranker.setGenerateRanking(true); // Opcional: para generar ranking completo

        AttributeSelection filter = new AttributeSelection();
        filter.setEvaluator(evaluator);
        filter.setSearch(ranker);

        return filter;
    }

    private static void saveDictionary(Instances data, String fileName) throws Exception {
        try (FileWriter fw = new FileWriter(fileName)) {
            // Excluimos el atributo de clase
            int classIndex = data.classIndex();

            for (int i = 0; i < data.numAttributes(); i++) {
                if (i != classIndex) {
                    fw.write(data.attribute(i).name() + "\n");
                }
            }
        }
        System.out.println("Diccionario con " + (data.numAttributes() - 1) + " términos guardado.");
    }

    public static void saveArff(Instances data, String fileName) throws Exception {
        ArffSaver saver = new ArffSaver();
        saver.setInstances(data);
        saver.setFile(new File(fileName));
        saver.writeBatch();
    }
}