package algo;

public final class Edge implements Comparable<Edge> {
    private int start;
    private int end;
    private double weight;
    private int seqID;

    public Edge(int start, int end, double weight) {
        this.start = start;
        this.end = end;
        this.weight = weight;
    }

    public Edge(int start, int end, double weight, int seqID) {
        this.start = start;
        this.end = end;
        this.weight = weight;
        this.seqID = seqID;
    }

    public int getStart() {
        return start;
    }

    public void setStart(int start) {
        this.start = start;
    }

    public int getEnd() {
        return end;
    }

    public void setEnd(int end) {
        this.end = end;
    }

    public double getWeight() {
        return weight;
    }

    public void setWeight(double weight) {
        this.weight = weight;
    }

    public int getSeqID() {
        return seqID;
    }

    public void setSeqID(int seqID) {
        this.seqID = seqID;
    }


    @Override
    public int hashCode() {
        return end;
    }

    @Override
    public boolean equals(Object o) {
        if (this == o) {
            return true;
        }
        if (o == null) {
            return false;
        }

        if (getClass() != o.getClass()) {
            return false;
        }

        Edge e = (Edge) o;
        if (start != e.start) {
            return false;
        } else if (end != e.end) {
            return false;
        } else if (weight != e.weight) {
            return false;
        }

        return true;
    }


    @Override
    public String toString() {
        return String.format("%d -> %d : %f", start, end, weight);
    }

    @Override
    public int compareTo(Edge edge) {
        int cmp = Double.compare(this.weight, edge.weight);

        return cmp == 0 ? Integer.compare(this.start, edge.start) : cmp;
    }


}
