# services/reasoning/scene_graph.py

import math
import networkx as nx


from services.detection.zones import DEFAULT_ZONES


class SceneGraphBuilder:
    def serialize_graph(self):

        serialized = []

        for source, target, data in self.graph.edges(data=True):

            relation = data.get("relation")

            edge_text = f"{source} -> [{relation}] -> {target}"

            if "distance" in data:

                edge_text += f" (distance={data['distance']})"

            serialized.append(edge_text)

        return "\n".join(serialized)

    def __init__(self, det_frame):

        self.det_frame = det_frame
        self.graph = nx.MultiDiGraph()

    def calculate_distance(self, center1, center2):

        x1, y1 = center1
        x2, y2 = center2

        return math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)

    def build_graph(self):

       

        for zone in DEFAULT_ZONES:

            self.graph.add_node(
                zone.name,
                type="zone"
            )

       

        detections = self.det_frame.detections

        for idx, det in enumerate(detections):

            node_name = f"{det.label}_{idx}"

            self.graph.add_node(
                node_name,
                type=det.label,
                center=det.center
            )

          

            for zone_name in det.zones_present:

                self.graph.add_edge(
                    node_name,
                    zone_name,
                    relation="INSIDE"
                )

        

        for i in range(len(detections)):

            for j in range(i + 1, len(detections)):

                det1 = detections[i]
                det2 = detections[j]

                node1 = f"{det1.label}_{i}"
                node2 = f"{det2.label}_{j}"

                dist = self.calculate_distance(
                    det1.center,
                    det2.center
                )

               

                if dist < 150:

                    self.graph.add_edge(
                        node1,
                        node2,
                        relation="NEAR",
                        distance=round(dist, 2)
                    )

               

                if (
                    det1.label == "person"
                    and det2.label in [
                        "backpack",
                        "handbag",
                        "cell phone",
                        "laptop"
                    ]
                    and dist < 80
                ):

                    self.graph.add_edge(
                        node1,
                        node2,
                        relation="HOLDING"
                    )

                

                if (
                    det1.label == "person"
                    and det2.label in [
                        "backpack",
                        "handbag",
                        "cell phone",
                        "laptop"
                    ]
                    and dist < 60
                ):

                    self.graph.add_edge(
                        node1,
                        node2,
                        relation="INTERACTING_WITH"
                    )

        return self.graph