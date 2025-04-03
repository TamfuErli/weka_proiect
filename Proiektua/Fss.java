package proiektua;
import weka.core.*;
import weka.attributeSelection.*;
import weka.filters.Filter;
import weka.filters.unsupervised.attribute.Remove;
import weka.core.converters.ArffSaver;

import java.io.File;
import java.io.FileReader;
import java.io.BufferedReader;

public class Fss {

    public static void main(String[] args) {
        try {
            // Cargar archivo ARFF
            String filePath = "tweets_bow.arff";  // Cambia la ruta según sea necesario
            BufferedReader reader = new BufferedReader(new FileReader(filePath));
            Instances data = new Instances(reader);
            reader.close();

            // Establecer el índice de la clase (si es necesario)
            data.setClassIndex(0); // Suponiendo que la última columna es la clase

            // Selección de atributos basada en InfoGain
            InfoGainAttributeEval eval = new InfoGainAttributeEval();
            Ranker search = new Ranker();
            search.setNumToSelect(2000); // Seleccionamos los 2000 atributos más importantes (ajusta según sea necesario)
            AttributeSelection attrSelect = new AttributeSelection();
            attrSelect.setEvaluator(eval);
            attrSelect.setSearch(search);

            // Realizar la selección de atributos
            attrSelect.SelectAttributes(data);

            // Obtener los atributos seleccionados
            int[] selectedAttributes = attrSelect.selectedAttributes();
            System.out.println("Atributos seleccionados: ");
            for (int idx : selectedAttributes) {
                System.out.print(idx + " ");
            }
            System.out.println();

            // Filtrar las instancias con los atributos seleccionados
            Remove remove = new Remove();
            StringBuilder indicesToRemove = new StringBuilder();
            for (int i = 0; i < data.numAttributes(); i++) {
                boolean isSelected = false;
                for (int selectedIdx : selectedAttributes) {
                    if (i == selectedIdx) {
                        isSelected = true;
                        break;
                    }
                }
                if (!isSelected) {
                    indicesToRemove.append(i + 1).append(",");
                }
            }
            System.out.println("Índices de atributos a eliminar: " + indicesToRemove.toString());

            // Eliminar atributos no seleccionados
            if (indicesToRemove.length() > 0) {
                indicesToRemove.setLength(indicesToRemove.length() - 1);  // Eliminar la última coma
                remove.setAttributeIndices(indicesToRemove.toString());
                remove.setInputFormat(data);
                Instances filteredData = Filter.useFilter(data, remove);

                // Guardar los datos filtrados en un archivo ARFF
                saveArff(filteredData, "tweets_bow_filtered.arff");
            }

        } catch (Exception e) {
            e.printStackTrace();
        }
    }

    // Método para guardar las instancias filtradas en un archivo ARFF
    public static void saveArff(Instances data, String fileName) throws Exception {
        ArffSaver saver = new ArffSaver();
        saver.setInstances(data);
        saver.setFile(new File(fileName));
        saver.writeBatch();  // Escribir las instancias en el archivo ARFF
        System.out.println("Archivo ARFF guardado en: " + fileName);
    }
}
