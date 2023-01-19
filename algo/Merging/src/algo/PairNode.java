package algo;

import java.io.Serializable;

public class PairNode implements Serializable {
    private static final long serialVersionUID = 7385160833386892234L;
    private int p1;
    private int p2;

    public PairNode(int p1, int p2) {
        this.p1 = p1;
        this.p2 = p2;
    }

    public int getP1() {
        return p1;
    }

    public void setP1(int p1) {
        this.p1 = p1;
    }

    public int getP2() {
        return p2;
    }

    public void setP2(int p2) {
        this.p2 = p2;
    }

    public boolean isValid() {
        if (this.p1 != this.p2) {
            return true;
        } else {
            return false;
        }

    }

    @Override
    public int hashCode() {
        int small = Math.min(p1, p2);
        int large = Math.max(p1, p2);
        return large * 4 + small;
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

        PairNode p = (PairNode) o;
        if (p1 == p.p1 && p2 == p.p2) {
            return true;
        } else if (p1 == p.p2 && p2 == p.p1) {
            return true;
        }

        return false;
    }

    @Override
    public String toString() {
        return String.format("%d -> %d", p1, p2);
    }
}
