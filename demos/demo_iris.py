import sys
from pathlib import Path

from AnyQt.QtWidgets import QApplication

import Orange.data
import Orange.distance

sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))

from owneighborjoining import OWNeighborJoining

def main():
    app = QApplication(sys.argv)
    data = Orange.data.Table("iris")
    dist = Orange.distance.Euclidean(data)
    dist.row_items = data
    dist.axis = 1

    w = OWNeighborJoining()
    w.set_distances(dist)
    w.handleNewSignals()
    w.show()

    sys.exit(app.exec())

if __name__ == "__main__":
    main()