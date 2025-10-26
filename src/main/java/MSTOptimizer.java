import java.io.*;
import java.util.*;
import com.google.gson.*;

// ============= MAIN CLASS =============
public class MSTOptimizer {
    public static void main(String[] args) {
        try {
            System.out.println("=== MST Transportation Network Optimizer ===\n");

            // Read input file
            Gson gson = new GsonBuilder().setPrettyPrinting().create();
            FileReader reader = new FileReader("ass_3_input.json");
            InputData inputData = gson.fromJson(reader, InputData.class);
            reader.close();

            System.out.println("Input file loaded successfully.");
            System.out.println("Processing " + inputData.graphs.size() + " graph(s)...\n");

            // Process all graphs
            OutputData outputData = new OutputData();
            outputData.results = new ArrayList<>();

            for (Graph graph : inputData.graphs) {
                System.out.println("Processing Graph " + graph.id + "...");
                Result result = processGraph(graph);
                outputData.results.add(result);
                System.out.println("Graph " + graph.id + " completed.\n");
            }

            // Write output file
            FileWriter writer = new FileWriter("ass_3_output.json");
            writer.write(gson.toJson(outputData));
            writer.close();

            System.out.println("Output file 'ass_3_output.json' created successfully.\n");

            // Display results
            displayResults(outputData);

        } catch (FileNotFoundException e) {
            System.err.println("Error: Input file 'ass_3_input.json' not found!");
            System.err.println("Please ensure the file is in the same directory as the program.");
        } catch (Exception e) {
            System.err.println("Error occurred: " + e.getMessage());
            e.printStackTrace();
        }
    }

    private static Result processGraph(Graph graph) {
        Result result = new Result();
        result.graph_id = graph.id;
        result.input_stats = new InputStats();
        result.input_stats.vertices = graph.nodes.size();
        result.input_stats.edges = graph.edges.size();

        // Run Prim's algorithm
        System.out.println("  Running Prim's algorithm...");
        PrimAlgorithm prim = new PrimAlgorithm(graph);
        result.prim = prim.findMST();

        // Run Kruskal's algorithm
        System.out.println("  Running Kruskal's algorithm...");
        KruskalAlgorithm kruskal = new KruskalAlgorithm(graph);
        result.kruskal = kruskal.findMST();

        return result;
    }

    private static void displayResults(OutputData outputData) {
        System.out.println("\n" + "=".repeat(60));
        System.out.println("                    RESULTS SUMMARY");
        System.out.println("=".repeat(60) + "\n");

        for (Result result : outputData.results) {
            System.out.println("┌─ Graph ID: " + result.graph_id + " " + "─".repeat(50));
            System.out.println("│ Input: " + result.input_stats.vertices + " vertices, " +
                    result.input_stats.edges + " edges");

            System.out.println("│");
            System.out.println("│ PRIM'S ALGORITHM:");
            System.out.println("│   Total Cost:      " + result.prim.total_cost);
            System.out.println("│   Operations:      " + result.prim.operations_count);
            System.out.println("│   Execution Time:  " +
                    String.format("%.2f", result.prim.execution_time_ms) + " ms");
            System.out.println("│   MST Edges:       " + formatEdges(result.prim.mst_edges));

            System.out.println("│");
            System.out.println("│ KRUSKAL'S ALGORITHM:");
            System.out.println("│   Total Cost:      " + result.kruskal.total_cost);
            System.out.println("│   Operations:      " + result.kruskal.operations_count);
            System.out.println("│   Execution Time:  " +
                    String.format("%.2f", result.kruskal.execution_time_ms) + " ms");
            System.out.println("│   MST Edges:       " + formatEdges(result.kruskal.mst_edges));

            System.out.println("│");
            System.out.println("│ COMPARISON:");
            double opDiff = ((double)(result.kruskal.operations_count - result.prim.operations_count) /
                    result.prim.operations_count) * 100;
            double timeDiff = ((result.kruskal.execution_time_ms - result.prim.execution_time_ms) /
                    result.prim.execution_time_ms) * 100;

            System.out.println("│   Cost Match:      " +
                    (result.prim.total_cost == result.kruskal.total_cost ? "✓ VERIFIED" : "✗ MISMATCH"));
            System.out.println("│   Operation Diff:  " + String.format("%+.1f%%", opDiff) +
                    " (Kruskal vs Prim)");
            System.out.println("│   Time Diff:       " + String.format("%+.1f%%", timeDiff) +
                    " (Kruskal vs Prim)");

            System.out.println("└" + "─".repeat(58) + "\n");
        }
    }

    private static String formatEdges(List<Edge> edges) {
        if (edges.isEmpty()) return "[]";
        StringBuilder sb = new StringBuilder();
        for (int i = 0; i < Math.min(3, edges.size()); i++) {
            if (i > 0) sb.append(", ");
            Edge e = edges.get(i);
            sb.append(e.from).append("-").append(e.to).append("(").append(e.weight).append(")");
        }
        if (edges.size() > 3) sb.append("...");
        return sb.toString();
    }
}

// ============= DATA CLASSES =============
class InputData {
    List<Graph> graphs;
}

class Graph {
    int id;
    List<String> nodes;
    List<Edge> edges;
}

class Edge {
    String from;
    String to;
    int weight;

    public Edge(String from, String to, int weight) {
        this.from = from;
        this.to = to;
        this.weight = weight;
    }

    @Override
    public String toString() {
        return from + "-" + to + "(" + weight + ")";
    }
}

class OutputData {
    List<Result> results;
}

class Result {
    int graph_id;
    InputStats input_stats;
    AlgorithmResult prim;
    AlgorithmResult kruskal;
}

class InputStats {
    int vertices;
    int edges;
}

class AlgorithmResult {
    List<Edge> mst_edges;
    int total_cost;
    int operations_count;
    double execution_time_ms;

    public AlgorithmResult() {
        mst_edges = new ArrayList<>();
    }
}

// ============= PRIM'S ALGORITHM =============
class PrimAlgorithm {
    private Graph graph;
    private Map<String, List<EdgeNode>> adjList;
    private int operationCount;

    static class EdgeNode implements Comparable<EdgeNode> {
        String node;
        int weight;
        String parent;

        EdgeNode(String node, int weight, String parent) {
            this.node = node;
            this.weight = weight;
            this.parent = parent;
        }

        @Override
        public int compareTo(EdgeNode other) {
            return Integer.compare(this.weight, other.weight);
        }
    }

    public PrimAlgorithm(Graph graph) {
        this.graph = graph;
        this.adjList = new HashMap<>();
        buildAdjacencyList();
    }

    private void buildAdjacencyList() {
        // Initialize adjacency list
        for (String node : graph.nodes) {
            adjList.put(node, new ArrayList<>());
        }

        // Add edges (undirected graph)
        for (Edge edge : graph.edges) {
            adjList.get(edge.from).add(new EdgeNode(edge.to, edge.weight, edge.from));
            adjList.get(edge.to).add(new EdgeNode(edge.from, edge.weight, edge.to));
        }
    }

    public AlgorithmResult findMST() {
        operationCount = 0;
        long startTime = System.nanoTime();

        AlgorithmResult result = new AlgorithmResult();
        Set<String> visited = new HashSet<>();
        PriorityQueue<EdgeNode> pq = new PriorityQueue<>();

        // Start from first node
        String startNode = graph.nodes.get(0);
        visited.add(startNode);
        operationCount++; // Add to visited set

        // Add all edges from start node to priority queue
        for (EdgeNode neighbor : adjList.get(startNode)) {
            pq.offer(neighbor);
            operationCount++; // Offer operation
        }

        // Main loop: build MST
        while (!pq.isEmpty() && visited.size() < graph.nodes.size()) {
            EdgeNode current = pq.poll();
            operationCount++; // Poll operation

            // Skip if node already visited
            if (visited.contains(current.node)) {
                operationCount++; // Contains check
                continue;
            }

            // Add node to visited set
            visited.add(current.node);
            operationCount++; // Add operation

            // Add edge to MST
            result.mst_edges.add(new Edge(current.parent, current.node, current.weight));
            result.total_cost += current.weight;
            operationCount += 2; // Add edge + cost update

            // Add all edges from current node
            for (EdgeNode neighbor : adjList.get(current.node)) {
                operationCount++; // Iteration
                if (!visited.contains(neighbor.node)) {
                    operationCount++; // Contains check
                    pq.offer(neighbor);
                    operationCount++; // Offer operation
                }
            }
        }

        long endTime = System.nanoTime();
        result.execution_time_ms = (endTime - startTime) / 1_000_000.0;
        result.operations_count = operationCount;

        return result;
    }
}

// ============= KRUSKAL'S ALGORITHM =============
class KruskalAlgorithm {
    private Graph graph;
    private int operationCount;

    public KruskalAlgorithm(Graph graph) {
        this.graph = graph;
    }

    public AlgorithmResult findMST() {
        operationCount = 0;
        long startTime = System.nanoTime();

        AlgorithmResult result = new AlgorithmResult();

        // Step 1: Sort edges by weight
        List<Edge> sortedEdges = new ArrayList<>(graph.edges);
        sortedEdges.sort(Comparator.comparingInt(e -> e.weight));
        // Count sort operations (n log n comparisons)
        operationCount += sortedEdges.size() * (int)(Math.log(sortedEdges.size()) / Math.log(2));

        // Step 2: Initialize Union-Find
        UnionFind uf = new UnionFind(graph.nodes);
        operationCount += graph.nodes.size(); // Make-set operations

        // Step 3: Process edges in sorted order
        for (Edge edge : sortedEdges) {
            operationCount++; // Iteration

            // Check if edge creates cycle
            String root1 = uf.find(edge.from);
            String root2 = uf.find(edge.to);
            operationCount += 2; // Two find operations

            if (!root1.equals(root2)) {
                operationCount++; // Comparison

                // Add edge to MST
                uf.union(edge.from, edge.to);
                operationCount++; // Union operation

                result.mst_edges.add(edge);
                result.total_cost += edge.weight;
                operationCount += 2; // Add edge + cost update

                // Early termination if MST complete
                if (result.mst_edges.size() == graph.nodes.size() - 1) {
                    operationCount++; // Size check
                    break;
                }
            }
        }

        long endTime = System.nanoTime();
        result.execution_time_ms = (endTime - startTime) / 1_000_000.0;
        result.operations_count = operationCount;

        return result;
    }

    // Union-Find (Disjoint Set) Data Structure
    static class UnionFind {
        private Map<String, String> parent;
        private Map<String, Integer> rank;

        public UnionFind(List<String> nodes) {
            parent = new HashMap<>();
            rank = new HashMap<>();

            // Make-set: each node is its own parent
            for (String node : nodes) {
                parent.put(node, node);
                rank.put(node, 0);
            }
        }

        // Find with path compression
        public String find(String node) {
            if (!parent.get(node).equals(node)) {
                parent.put(node, find(parent.get(node))); // Path compression
            }
            return parent.get(node);
        }

        // Union by rank
        public void union(String node1, String node2) {
            String root1 = find(node1);
            String root2 = find(node2);

            if (!root1.equals(root2)) {
                int rank1 = rank.get(root1);
                int rank2 = rank.get(root2);

                // Attach smaller rank tree under root of higher rank tree
                if (rank1 < rank2) {
                    parent.put(root1, root2);
                } else if (rank1 > rank2) {
                    parent.put(root2, root1);
                } else {
                    parent.put(root2, root1);
                    rank.put(root1, rank1 + 1);
                }
            }
        }
    }
}