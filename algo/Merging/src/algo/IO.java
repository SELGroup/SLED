package algo;

import java.io.BufferedReader;
import java.io.FileInputStream;
import java.io.InputStreamReader;
import java.util.HashMap;
import java.util.Set;
import java.util.TreeSet;

public class IO {
    /**
     * 从指定指定获取无向图
     * 注意文件的格式：
     * 第一行为节点的总数
     * 从第二行起，每一行为一个边：
     * 例如：1 2 3.44445
     * 中间用空格隔开
     *
     * @param filePath
     * @throws Exception
     */
    public static Graph getUndirGraphFromFile(String filePath) throws Exception {
        FileInputStream file = new FileInputStream(filePath);
        BufferedReader bf = new BufferedReader(new InputStreamReader(file));

        int numNodes = Integer.parseInt(bf.readLine());
        Graph g = new Graph(numNodes);
        double sumDegrees = 0.0;
        HashMap<PairNode, Double> weights = g.getWeights();
        HashMap<Integer, Set<Integer>> connection = g.getConnection();
        double[] nodeDegree = g.getNodeDegree();

        String line;
        while ((line = bf.readLine()) != null) {
            String[] edge = line.trim().split(" ");
            int start = Integer.parseInt(edge[0]);
            int end = Integer.parseInt(edge[1]);
            double weight = Double.parseDouble(edge[2]);
            PairNode pair = new PairNode(start, end);
            if (!valid(pair, weight)) {
                continue;
            }

            if (!weights.containsKey(pair)) {
                weights.put(pair, weight);
                putConnection(connection, start, end);
                nodeDegree[start] += weight;
                nodeDegree[end] += weight;
                sumDegrees += 2 * weight;
            }


        }

        g.setSumDegrees(sumDegrees);
        file.close();

        return g;
    }

    private static boolean valid(PairNode pair, double weight) {
        if (!pair.isValid()) {
            System.out.println("edge is illegal, a node cannot connected with itself");
            return false;
        } else if (weight == 0) {
            System.out.println("edge break");
            return false;
        }

        return true;
    }


    /**
     *
     * @param connection
     * @param start
     * @param end
     */
    public static void putConnection(HashMap<Integer, Set<Integer>> connection, int start, int end) {
        if (!connection.containsKey(start)) {
            connection.put(start, new TreeSet<Integer>() {{
                add(end);
            }});
        } else {
            connection.get(start).add(end);
        }
        //start end交换位置
        if (!connection.containsKey(end)) {
            connection.put(end, new TreeSet<Integer>() {{
                add(start);
            }});
        } else {
            connection.get(end).add(start);
        }
    }


}
