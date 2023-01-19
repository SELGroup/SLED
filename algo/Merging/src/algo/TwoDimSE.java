package algo;


import java.io.*;
import java.math.BigDecimal;
import java.util.HashMap;
import java.util.Set;
import java.util.TreeSet;


public class TwoDimSE {
    private final double sumDegrees;
    private double oneDimSE;
    private double twoDimSE;
    private double compressionRatio;
    private final HashMap<Integer, Set<Integer>> communities;
    private final double[] volumes;
    private final double[] gs;
    private final HashMap<PairNode, Double> cuts;
    private final HashMap<Integer, Set<Integer>> connections;
    private final HashMap<PairNode, CommDeltaH> commDeltaHMap;
    private final TreeSet<CommDeltaH> commDeltaHSet;

    public TwoDimSE(Graph graph) {
        this.oneDimSE = 0.0;
        this.twoDimSE = 0.0;

        this.sumDegrees = graph.getSumDegrees();
        int initialCap = 3 * graph.getNumNodes() / 4 + 1;
        this.volumes = graph.getNodeDegree();
        this.gs = graph.getNodeDegree().clone();
        this.communities = new HashMap<>(initialCap);
        this.cuts = graph.getWeights();
        this.connections = graph.getConnection();
        this.commDeltaHMap = new HashMap<>();
        this.commDeltaHSet = new TreeSet<>();
    }


    private void min2dSE(String saveFilePath, boolean doPrintNDI, boolean doSave) throws IOException {
        initEncodingTree();
        twoDimSE = oneDimSE;
        CommDeltaH maxCommDeltaH = commDeltaHSet.last();

        //Merge module pair with maximum deltaH, util no such module pair can be found.
        while (maxCommDeltaH.getDeltaH() > 0 && !commDeltaHSet.isEmpty()) {
            PairNode comms = maxCommDeltaH.getPairComms();
            double deltaH = maxCommDeltaH.getDeltaH();
            twoDimSE -= deltaH;
            updateCommunities(maxCommDeltaH);
            maxCommDeltaH = commDeltaHSet.last();
        }

    }

    public void min2dSE(boolean doPrintNDI) {
        try {
            min2dSE(" ",  doPrintNDI, false);
        } catch (IOException e) {
            e.printStackTrace();
        }
    }

    public void min2dSE(String saveFilePath, boolean doPrintNDI) throws IOException {
        min2dSE(saveFilePath, doPrintNDI, true);
    }


    // Update information of modules after merging module pairs.
    private void updateCommunities(CommDeltaH commDeltaH) {
        PairNode comms = commDeltaH.getPairComms();
        double deltaH = commDeltaH.getDeltaH();
        int commLeft = comms.getP1();
        int commRight = comms.getP2();

        double vi = volumes[commLeft];
        double gi = gs[commLeft];
        double vj = volumes[commRight];
        double gj = gs[commRight];
        volumes[commLeft] = vi + vj;
        gs[commLeft] = gi + gj - 2 * cuts.get(comms);
        volumes[commRight] = 0.0;
        gs[commRight] = 0.0;

        communities.get(commLeft).addAll(communities.get(commRight));
        communities.remove(commRight);
        commDeltaHMap.remove(comms);
        commDeltaHSet.remove(commDeltaH);
        connections.get(commLeft).remove(commRight);
        connections.get(commRight).remove(commLeft);
        cuts.remove(comms);

        updateCutAndDeltaH(commLeft, commRight);

    }


    private void updateCutAndDeltaH(int commLeft, int commRight) {
        Set<Integer> connLeft = connections.get(commLeft);
        Set<Integer> connRight = connections.get(commRight);

        double Vi = volumes[commLeft];
        double Gi = gs[commLeft];
        double Gk;
        double Vk;
        double Gx;
        double newDelta;
        //Traversal modules connecting to commLeft.
        for (int k : connLeft) {
            double cutIk;
            PairNode pairLeftAndK = new PairNode(commLeft, k);
            if (connRight.contains(k)) {    //if module k connect to both commLeft and commRight.
                PairNode pairRightAndK = new PairNode(commRight, k);
                cutIk = cuts.get(pairLeftAndK) + cuts.get(pairRightAndK);
                connRight.remove(k);
//                commDeltaHSet.remove(new CommDeltaH(pairRightAndK, commDeltaHMap.get(pairRightAndK)));
                commDeltaHSet.remove(commDeltaHMap.get(pairRightAndK));
                commDeltaHMap.remove(pairRightAndK);
                cuts.remove(pairRightAndK);
                connections.get(k).remove(commRight);
            } else {
                cutIk = cuts.get(pairLeftAndK);
            }
            Gk = gs[k];
            Vk = volumes[k];
            Gx = Gi + Gk - 2 * cutIk;
            newDelta = computeDeltaH(Vi, Vk, Gi, Gk, Gx, sumDegrees);

            cuts.put(pairLeftAndK, cutIk);
            commDeltaHSet.remove(commDeltaHMap.get(pairLeftAndK));
            CommDeltaH newDeltaH = new CommDeltaH(pairLeftAndK, newDelta);
            commDeltaHSet.add(newDeltaH);
            commDeltaHMap.put(pairLeftAndK, newDeltaH);

        }
        //traversal modules connecting to commRight but not to commLeft.
        for (int k : connRight) {
            PairNode pairRightAndK = new PairNode(commRight, k);
            double cutJk = cuts.get(pairRightAndK);
            Vk = volumes[k];
            Gk = gs[k];
            Gx = Gi + Gk - 2 * cutJk;
            newDelta = computeDeltaH(Vi, Vk, Gi, Gk, Gx, sumDegrees);

            PairNode pairLeftAndK = new PairNode(commLeft, k);
            cuts.put(pairLeftAndK, cutJk);
            cuts.remove(pairRightAndK);
            CommDeltaH commDeltaH = new CommDeltaH(pairLeftAndK, newDelta);
            commDeltaHSet.remove(commDeltaHMap.get(pairRightAndK));
            commDeltaHSet.add(commDeltaH);
            commDeltaHMap.remove(pairRightAndK);
            commDeltaHMap.put(pairLeftAndK, commDeltaH);
            connections.get(commLeft).add(k);
            connections.get(k).add(commLeft);
            connections.get(k).remove(commRight);
        }

        connRight.clear();
    }

    /**
     * Initiate encoding tree with one root tree and all graph nodes as tree leaves.
     */
    private void initEncodingTree() {
        for (PairNode p : cuts.keySet()) {
            double vi = volumes[p.getP1()];
            double vj = volumes[p.getP2()];
            double gi = vi;
            double gj = vj;
            double gx = vi + vj - 2 * cuts.get(p);
            double deltaH = computeDeltaH(vi, vj, gi, gj, gx, sumDegrees);
            CommDeltaH commDeltaH = new CommDeltaH(p, deltaH);
            commDeltaHMap.put(p, commDeltaH);
            commDeltaHSet.add(commDeltaH);
        }

        for (int i = 1; i < volumes.length; i++) {
            if (volumes[i] > 0.0) {
                int finalI = i;
                communities.put(i, new TreeSet<Integer>() {{
                    add(finalI);
                }});
                oneDimSE -= (volumes[i] / sumDegrees) * (Math.log(volumes[i] / sumDegrees)) / Math.log(2);
            }
        }
    }

    private void seInfo() {
        this.compressionRatio = (oneDimSE - twoDimSE) / oneDimSE;
        System.out.printf("The One and Two dimension SE: %f, %f\nDecoding Information(Compression Information) : %f%n",
                oneDimSE, twoDimSE, oneDimSE - twoDimSE);
        System.out.printf("The Normalized Decoding Information(Compression Ratio) is %f%n", compressionRatio);
    }

    public void saveResult(String fileName) throws IOException {
        BufferedWriter bw = new BufferedWriter(new FileWriter(fileName));
        for (Set<Integer> res : communities.values()) {
            for (int i : res) {
//                System.out.print(i + "\t");
                bw.write(i + "\t");
            }
//            System.out.println();
            bw.write("\n");
        }

        bw.close();
    }

    public HashMap<Integer, Set<Integer>> getCommunities() {
        return communities;
    }

    public double getCompressionRatio(){
        return compressionRatio;
    }

    public void savePartitionResult(String saveFileName) {
        try {
            ObjectOutputStream oos = new ObjectOutputStream(new FileOutputStream(saveFileName));
            oos.writeObject(communities);
            oos.close();
        } catch (IOException e) {
            e.printStackTrace();
        }
    }

    public HashMap<Integer, Set<Integer>> readPartitionResult(String fileName) {
        HashMap<Integer, Set<Integer>> res = new HashMap<>();
        try {
            ObjectInputStream ois = new ObjectInputStream(new FileInputStream(fileName));
            res = (HashMap<Integer, Set<Integer>>) ois.readObject();
            ois.close();
        } catch (IOException | ClassNotFoundException e) {
            e.printStackTrace();
        }
        return res;
    }

    private double computeDeltaH(double vi, double vj, double gi, double gj, double gx, double sumDegrees) {
        BigDecimal a1 = new BigDecimal(vi * (Math.log(vi) / (Math.log(2) + 0.0)));
        BigDecimal a2 = new BigDecimal(vj * (Math.log(vj) / (Math.log(2) + 0.0)));
        BigDecimal a3 = new BigDecimal((vi + vj) * (Math.log(vi + vj) / (Math.log(2) + 0.0)));
        BigDecimal a4 = new BigDecimal(gi * (Math.log(vi / (sumDegrees + 0.0)) / Math.log(2)));
        BigDecimal a5 = new BigDecimal(gj * (Math.log(vj / (sumDegrees + 0.0)) / Math.log(2)));
        BigDecimal a6 = new BigDecimal(gx * (Math.log((vi + vj) / (sumDegrees + 0.0)) / Math.log(2)));
//        System.out.println(String.format("a1, a2, a3, a4, a5, a6: %f, %f, %f, %f, %f, %f", a1, a2, a3, a4, a5, a6));
        BigDecimal b1 = a1.add(a2);
        BigDecimal b2 = b1.subtract(a3);
        BigDecimal b3 = b2.subtract(a4);
        BigDecimal b4 = b3.subtract(a5);
        BigDecimal b5 = b4.add(a6);
        return b5.doubleValue() / (sumDegrees + 0.0);
    }
    

}
