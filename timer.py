
>>> from PyQt4.QtCore import *

>>> timer = QTimer()

>>> timer.setInterval(1000)
>>> QObject.connect(timer, SIGNAL("timeout()"),
qgis.utils.iface.mapCanvas().refresh)
>>> timer.start()
