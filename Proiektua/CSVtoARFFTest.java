package Proiektua;

import java.io.*;
import java.util.ArrayList;
import java.util.List;

public class CSVtoARFFTest {

    public static void convert(String inputFile, String cleanedCSV, String outputARFF) throws IOException {
        // Paso 1: Limpiar el archivo CSV
        cleanCSV(inputFile, cleanedCSV);

        // Paso 2: Convertir a ARFF
        convertToARFF(cleanedCSV, outputARFF);
    }

    private static void cleanCSV(String inputPath, String outputPath) throws IOException {
        try (BufferedReader reader = new BufferedReader(new FileReader(inputPath));
             BufferedWriter writer = new BufferedWriter(new FileWriter(outputPath))) {

            String line;
            StringBuilder recordBuffer = new StringBuilder();
            boolean firstLine = true;
            int expectedColumns = 5;
            int quoteCount = 0;

            while ((line = reader.readLine()) != null || recordBuffer.length() > 0) {
                if (line != null) {
                    quoteCount += countQuotes(line);
                    recordBuffer.append(line).append("\n");
                }

                if (quoteCount % 2 == 0 || line == null) {
                    String completeRecord = recordBuffer.toString().trim();
                    if (!completeRecord.isEmpty()) {
                        List<String> fields = parseCSVLine(completeRecord);

                        if (firstLine) {
                            writer.write(String.join(",", fields));
                            writer.newLine();
                            firstLine = false;
                        } else if (fields.size() == expectedColumns) {
                            List<String> cleanedFields = new ArrayList<>();
                            for (String field : fields) {
                                cleanedFields.add(cleanField(field));
                            }
                            writer.write(String.join(",", cleanedFields));
                            writer.newLine();
                        }
                    }
                    recordBuffer = new StringBuilder();
                    quoteCount = 0;
                }
            }
        }
    }

    private static String cleanField(String field) {
        String cleaned = field.trim();
        if (cleaned.startsWith("\"") && cleaned.endsWith("\"")) {
            cleaned = cleaned.substring(1, cleaned.length() - 1);
        }
        cleaned = cleaned.replace("\"\"", "\"")
                .replaceAll("[\\r\\n]+", " ")
                .replaceAll("\\s+", " ")
                .trim();
        return "\"" + cleaned.replace("\"", "\"\"") + "\"";
    }

    private static int countQuotes(String str) {
        int count = 0;
        for (int i = 0; i < str.length(); i++) {
            if (str.charAt(i) == '"') {
                count++;
            }
        }
        return count;
    }

    private static void convertToARFF(String inputCSV, String outputARFF) throws IOException {
        try (BufferedReader reader = new BufferedReader(new FileReader(inputCSV));
             BufferedWriter writer = new BufferedWriter(new FileWriter(outputARFF))) {

            writer.write("@RELATION tweets\n\n");
            writer.write("@ATTRIBUTE Topic {twitter,google,apple,microsoft}\n");
            writer.write("@ATTRIBUTE Sentiment {positive,negative,neutral,irrelevant}\n");
            writer.write("@ATTRIBUTE TweetId string\n");
            writer.write("@ATTRIBUTE TweetDate string\n");
            writer.write("@ATTRIBUTE TweetText string\n\n");
            writer.write("@DATA\n");

            String line;
            boolean firstLine = true;

            while ((line = reader.readLine()) != null) {
                if (firstLine) {
                    firstLine = false;
                    continue;
                }
                if (!line.trim().isEmpty()) {
                    writer.write(line);
                    writer.newLine();
                }
            }
        }
    }

    private static List<String> parseCSVLine(String line) {
        List<String> fields = new ArrayList<>();
        StringBuilder currentField = new StringBuilder();
        boolean inQuotes = false;

        for (int i = 0; i < line.length(); i++) {
            char c = line.charAt(i);

            if (c == '"') {
                if (inQuotes && i < line.length() - 1 && line.charAt(i + 1) == '"') {
                    currentField.append('"');
                    i++;
                } else {
                    inQuotes = !inQuotes;
                }
            } else if (c == ',' && !inQuotes) {
                fields.add(currentField.toString());
                currentField = new StringBuilder();
            } else {
                currentField.append(c);
            }
        }

        fields.add(currentField.toString());
        return fields;
    }
}