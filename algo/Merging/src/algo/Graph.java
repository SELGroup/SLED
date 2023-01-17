package algo;

import java.io.*;
import java.util.HashMap;
import java.util.Set;

public class Graph implements Serializable {
    private static final long serialVersionUID = -4217430621824112183L;
    private int numNodes;
    private double sumDegrees;

    private HashMap<PairNode, Double> weights;
    private HashMap<Integer, Set<Integer>> connection;
    //此处使用数组而非HashMap的原因是：图的节点的编号连续，如从1-55555，共55555个节点
    private double[] nodeDegree;


    public Graph(int numNodes) {
        this.numNodes = numNodes;
        int initialCap = 3 * numNodes / 4 + 1;
        this.weights = new HashMap<>(initialCap);
        this.connection = new HashMap<>(initialCap);
        this.nodeDegree = new double[numNodes + 1]; //节点从1开始而非0
    }

    public int getNumNodes() {
        return numNodes;
    }

    public void setNumNodes(int numNodes) {
        this.numNodes = numNodes;
    }

    public double getSumDegrees() {
        return sumDegrees;
    }

    public void setSumDegrees(double sumDegrees) {
        this.sumDegrees = sumDegrees;
    }

    public HashMap<PairNode, Double> getWeights() {
        return weights;
    }

    public void setWeights(HashMap<PairNode, Double> weights) {
        this.weights = weights;
    }

    public double[] getNodeDegree() {
        return nodeDegree;
    }

    public void setNodeDegree(double[] nodeDegree) {
        this.nodeDegree = nodeDegree;
    }

    public HashMap<Integer, Set<Integer>> getConnection() {
        return connection;
    }

    public void setConnection(HashMap<Integer, Set<Integer>> connection) {
        this.connection = connection;
    }

    /**
     * java原生的序列化和反序列化太耗时
     * @param fileName
     */
    @Deprecated
    public void write2File(String fileName) {
        try {
            ObjectOutputStream oos = new ObjectOutputStream(new FileOutputStream(fileName));
            oos.writeObject(this);
            oos.close();
        } catch (IOException e) {
            e.printStackTrace();
        }
    }
    @Deprecated
    public static void write2File(Graph graph, String fileName) {
        try {
            ObjectOutputStream oos = new ObjectOutputStream(new FileOutputStream(fileName));
            oos.writeObject(graph);
            oos.close();
        } catch (IOException e) {
            e.printStackTrace();
        }
    }

    public static Graph readFromFile(String fileName) {
        Graph g = null;
        try {
            ObjectInputStream ois = new ObjectInputStream(new FileInputStream(fileName));
            g = (Graph) ois.readObject();
            ois.close();
        } catch (IOException | ClassNotFoundException e) {
            e.printStackTrace();
        }

        return g;
    }
}

