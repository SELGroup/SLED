package algo;

import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.FileInputStream;
import java.io.FileWriter;
import java.io.InputStreamReader;
import java.util.HashMap;
import java.util.Scanner;
import java.util.Set;

public class Merging {
    public static Graph constructGraph(String graphPath) throws Exception {
        Scanner scan = new Scanner(new BufferedReader(new InputStreamReader(new FileInputStream(graphPath), "UTF-8")));
        int numNodes = Integer.parseInt(scan.nextLine().strip());
        Graph g = new Graph(numNodes);
        double sumDegrees = g.getSumDegrees();
        HashMap<PairNode, Double> weights = g.getWeights();
        HashMap<Integer, Set<Integer>> connection = g.getConnection();
        double[] nodeDegree = g.getNodeDegree();
        while(scan.hasNextLine()) {
            String[] line = scan.nextLine().strip().split("\t");
            if(line.length < 3)
                continue;
            int start = Integer.parseInt(line[0]);
            int end = Integer.parseInt(line[1]);
            if(start == end)
                continue;
            double weight = Double.parseDouble(line[2]);

            PairNode pair = new PairNode(start, end);
            if (!weights.containsKey(pair)) {
                weights.put(pair, weight);
                IO.putConnection(connection, start, end);
                nodeDegree[start] += weight;
                nodeDegree[end] += weight;
                sumDegrees += 2 * weight;
            }
        }
        scan.close();
        g.setSumDegrees(sumDegrees);
        return g;
    }

    public static void main(String[] args) throws Exception {
        String graphPath = args[0];
        String partitionPath = args[1];

        Graph g = constructGraph(graphPath);
        TwoDimSE algo = new TwoDimSE(g);
        algo.min2dSE(false);
        HashMap<Integer, Set<Integer>> partition = algo.getCommunities();
        BufferedWriter bw = new BufferedWriter(new FileWriter(partitionPath));
        for (int key : partition.keySet()) {
            for (int nodei : partition.get(key)) {
                bw.write(nodei + "\t");
            }
            bw.write("\n");
        }
        bw.close();
    }
}
