from typing import Dict, List

from jaxite.jaxite_bool import jaxite_bool
from jaxite.jaxite_lib import types


def redact_ssn(
    my_string: List[types.LweCiphertext],
    sks: jaxite_bool.ServerKeySet,
    params: jaxite_bool.Parameters,
) -> List[types.LweCiphertext]:
  temp_nodes: Dict[int, types.LweCiphertext] = {}
  out = [None] * 256
  inputs = [
      (my_string[10], my_string[9], jaxite_bool.constant(False, params), 1),
      (my_string[11], my_string[12], my_string[13], 128),
      (my_string[3], my_string[4], my_string[5], 128),
  ]
  outputs = jaxite_bool.pmap_lut3(inputs, sks, params)
  temp_nodes[1645] = outputs[0]
  temp_nodes[1646] = outputs[1]
  temp_nodes[1650] = outputs[2]
  inputs = [
      (my_string[18], my_string[17], jaxite_bool.constant(False, params), 1),
      (temp_nodes[1646], temp_nodes[1645], my_string[14], 13),
      (my_string[12], my_string[13], jaxite_bool.constant(False, params), 8),
      (my_string[1], my_string[2], temp_nodes[1650], 224),
  ]
  outputs = jaxite_bool.pmap_lut3(inputs, sks, params)
  temp_nodes[1641] = outputs[0]
  temp_nodes[1644] = outputs[1]
  temp_nodes[1647] = outputs[2]
  temp_nodes[1649] = outputs[3]
  inputs = [
      (my_string[4], my_string[5], jaxite_bool.constant(False, params), 8),
      (my_string[30], my_string[31], jaxite_bool.constant(False, params), 1),
      (my_string[34], my_string[33], my_string[38], 1),
      (my_string[35], my_string[36], my_string[37], 128),
  ]
  outputs = jaxite_bool.pmap_lut3(inputs, sks, params)
  temp_nodes[1652] = outputs[0]
  temp_nodes[1655] = outputs[1]
  temp_nodes[1665] = outputs[2]
  temp_nodes[1666] = outputs[3]
  inputs = [
      (my_string[58], my_string[57], jaxite_bool.constant(False, params), 1),
      (my_string[19], temp_nodes[1641], my_string[22], 13),
      (my_string[23], my_string[20], my_string[21], 64),
      (my_string[15], temp_nodes[1647], temp_nodes[1644], 64),
  ]
  outputs = jaxite_bool.pmap_lut3(inputs, sks, params)
  temp_nodes[1634] = outputs[0]
  temp_nodes[1639] = outputs[1]
  temp_nodes[1640] = outputs[2]
  temp_nodes[1643] = outputs[3]
  inputs = [
      (temp_nodes[1649], my_string[6], jaxite_bool.constant(False, params), 1),
      (my_string[7], temp_nodes[1652], jaxite_bool.constant(False, params), 4),
      (temp_nodes[1655], my_string[28], my_string[29], 128),
      (my_string[25], my_string[26], my_string[27], 31),
  ]
  outputs = jaxite_bool.pmap_lut3(inputs, sks, params)
  temp_nodes[1648] = outputs[0]
  temp_nodes[1651] = outputs[1]
  temp_nodes[1654] = outputs[2]
  temp_nodes[1656] = outputs[3]
  inputs = [
      (my_string[41], my_string[42], my_string[46], 1),
      (my_string[43], my_string[44], my_string[45], 128),
      (my_string[38], temp_nodes[1666], temp_nodes[1665], 241),
      (my_string[37], my_string[36], my_string[38], 7),
  ]
  outputs = jaxite_bool.pmap_lut3(inputs, sks, params)
  temp_nodes[1659] = outputs[0]
  temp_nodes[1661] = outputs[1]
  temp_nodes[1664] = outputs[2]
  temp_nodes[1667] = outputs[3]
  inputs = [
      (my_string[50], my_string[49], jaxite_bool.constant(False, params), 1),
  ]
  outputs = jaxite_bool.pmap_lut3(inputs, sks, params)
  temp_nodes[1670] = outputs[0]
  inputs = [
      (my_string[61], my_string[60], my_string[62], 7),
      (
          temp_nodes[1639],
          temp_nodes[1640],
          jaxite_bool.constant(False, params),
          8,
      ),
      (temp_nodes[1651], temp_nodes[1648], temp_nodes[1643], 112),
      (
          temp_nodes[1654],
          temp_nodes[1656],
          jaxite_bool.constant(False, params),
          8,
      ),
  ]
  outputs = jaxite_bool.pmap_lut3(inputs, sks, params)
  temp_nodes[1635] = outputs[0]
  temp_nodes[1638] = outputs[1]
  temp_nodes[1642] = outputs[2]
  temp_nodes[1653] = outputs[3]
  inputs = [
      (my_string[46], temp_nodes[1661], temp_nodes[1659], 241),
      (my_string[45], my_string[44], my_string[46], 7),
      (temp_nodes[1667], my_string[39], temp_nodes[1664], 16),
      (my_string[53], my_string[52], my_string[54], 7),
  ]
  outputs = jaxite_bool.pmap_lut3(inputs, sks, params)
  temp_nodes[1658] = outputs[0]
  temp_nodes[1662] = outputs[1]
  temp_nodes[1663] = outputs[2]
  temp_nodes[1671] = outputs[3]
  inputs = [
      (my_string[68], my_string[69], jaxite_bool.constant(False, params), 8),
      (my_string[66], my_string[65], jaxite_bool.constant(False, params), 1),
      (temp_nodes[1634], my_string[59], my_string[62], 244),
      (temp_nodes[1670], my_string[51], my_string[54], 244),
  ]
  outputs = jaxite_bool.pmap_lut3(inputs, sks, params)
  temp_nodes[1673] = outputs[0]
  temp_nodes[1675] = outputs[1]
  temp_nodes[1479] = outputs[2]
  temp_nodes[1481] = outputs[3]
  inputs = [
      (temp_nodes[1642], temp_nodes[1638], temp_nodes[1653], 176),
      (temp_nodes[1662], my_string[47], temp_nodes[1658], 16),
      (temp_nodes[1673], my_string[70], jaxite_bool.constant(False, params), 1),
      (my_string[74], my_string[73], jaxite_bool.constant(False, params), 1),
  ]
  outputs = jaxite_bool.pmap_lut3(inputs, sks, params)
  temp_nodes[1637] = outputs[0]
  temp_nodes[1657] = outputs[1]
  temp_nodes[1676] = outputs[2]
  temp_nodes[1680] = outputs[3]
  inputs = [
      (
          temp_nodes[1638],
          temp_nodes[1643],
          jaxite_bool.constant(False, params),
          8,
      ),
      (
          temp_nodes[1663],
          temp_nodes[1653],
          jaxite_bool.constant(False, params),
          8,
      ),
      (my_string[63], temp_nodes[1635], temp_nodes[1479], 16),
      (my_string[55], temp_nodes[1671], temp_nodes[1481], 16),
  ]
  outputs = jaxite_bool.pmap_lut3(inputs, sks, params)
  temp_nodes[1720] = outputs[0]
  temp_nodes[1721] = outputs[1]
  temp_nodes[1480] = outputs[2]
  temp_nodes[1482] = outputs[3]
  inputs = [
      (temp_nodes[1675], my_string[67], my_string[70], 244),
  ]
  outputs = jaxite_bool.pmap_lut3(inputs, sks, params)
  temp_nodes[1483] = outputs[0]
  inputs = [
      (temp_nodes[1480], temp_nodes[1635], my_string[63], 1),
      (temp_nodes[1663], temp_nodes[1637], temp_nodes[1657], 208),
      (temp_nodes[1482], temp_nodes[1671], my_string[55], 1),
      (my_string[77], my_string[76], my_string[78], 7),
  ]
  outputs = jaxite_bool.pmap_lut3(inputs, sks, params)
  temp_nodes[1632] = outputs[0]
  temp_nodes[1636] = outputs[1]
  temp_nodes[1668] = outputs[2]
  temp_nodes[1681] = outputs[3]
  inputs = [
      (my_string[84], my_string[85], jaxite_bool.constant(False, params), 8),
      (my_string[82], my_string[81], jaxite_bool.constant(False, params), 1),
      (temp_nodes[1721], temp_nodes[1720], temp_nodes[1657], 208),
      (my_string[71], temp_nodes[1676], temp_nodes[1483], 16),
  ]
  outputs = jaxite_bool.pmap_lut3(inputs, sks, params)
  temp_nodes[1683] = outputs[0]
  temp_nodes[1685] = outputs[1]
  temp_nodes[1756] = outputs[2]
  temp_nodes[1484] = outputs[3]
  inputs = [
      (temp_nodes[1680], my_string[75], my_string[78], 244),
  ]
  outputs = jaxite_bool.pmap_lut3(inputs, sks, params)
  temp_nodes[1485] = outputs[0]
  inputs = [
      (temp_nodes[1668], temp_nodes[1636], temp_nodes[1632], 208),
      (temp_nodes[1484], temp_nodes[1676], my_string[71], 1),
      (temp_nodes[1683], my_string[86], jaxite_bool.constant(False, params), 1),
      (my_string[90], my_string[89], jaxite_bool.constant(False, params), 1),
  ]
  outputs = jaxite_bool.pmap_lut3(inputs, sks, params)
  temp_nodes[1631] = outputs[0]
  temp_nodes[1672] = outputs[1]
  temp_nodes[1686] = outputs[2]
  temp_nodes[1690] = outputs[3]
  inputs = [
      (
          temp_nodes[1636],
          temp_nodes[1756],
          jaxite_bool.constant(False, params),
          6,
      ),
      (
          temp_nodes[1632],
          temp_nodes[1668],
          jaxite_bool.constant(False, params),
          8,
      ),
      (temp_nodes[1636], temp_nodes[1668], temp_nodes[1756], 64),
      (
          temp_nodes[1657],
          temp_nodes[1721],
          jaxite_bool.constant(False, params),
          8,
      ),
  ]
  outputs = jaxite_bool.pmap_lut3(inputs, sks, params)
  temp_nodes[1719] = outputs[0]
  temp_nodes[1722] = outputs[1]
  temp_nodes[1755] = outputs[2]
  temp_nodes[1757] = outputs[3]
  inputs = [
      (my_string[79], temp_nodes[1681], temp_nodes[1485], 16),
      (temp_nodes[1685], my_string[83], my_string[86], 244),
  ]
  outputs = jaxite_bool.pmap_lut3(inputs, sks, params)
  temp_nodes[1486] = outputs[0]
  temp_nodes[1487] = outputs[1]
  inputs = [
      (
          temp_nodes[1631],
          temp_nodes[1672],
          jaxite_bool.constant(False, params),
          4,
      ),
      (temp_nodes[1486], temp_nodes[1681], my_string[79], 1),
      (my_string[93], my_string[92], my_string[94], 7),
      (my_string[97], my_string[98], jaxite_bool.constant(False, params), 1),
  ]
  outputs = jaxite_bool.pmap_lut3(inputs, sks, params)
  temp_nodes[1630] = outputs[0]
  temp_nodes[1677] = outputs[1]
  temp_nodes[1691] = outputs[2]
  temp_nodes[1695] = outputs[3]
  inputs = [
      (
          temp_nodes[1719],
          temp_nodes[1722],
          jaxite_bool.constant(False, params),
          4,
      ),
      (temp_nodes[1719], temp_nodes[1722], temp_nodes[1636], 64),
      (temp_nodes[1668], temp_nodes[1755], temp_nodes[1757], 44),
      (my_string[87], temp_nodes[1686], temp_nodes[1487], 16),
  ]
  outputs = jaxite_bool.pmap_lut3(inputs, sks, params)
  temp_nodes[1718] = outputs[0]
  temp_nodes[1748] = outputs[1]
  temp_nodes[1754] = outputs[2]
  temp_nodes[1488] = outputs[3]
  inputs = [
      (temp_nodes[1690], my_string[91], my_string[94], 244),
  ]
  outputs = jaxite_bool.pmap_lut3(inputs, sks, params)
  temp_nodes[1489] = outputs[0]
  inputs = [
      (
          temp_nodes[1630],
          temp_nodes[1677],
          jaxite_bool.constant(False, params),
          4,
      ),
      (temp_nodes[1488], temp_nodes[1686], my_string[87], 1),
      (my_string[101], my_string[100], my_string[102], 7),
      (my_string[108], my_string[109], jaxite_bool.constant(False, params), 8),
  ]
  outputs = jaxite_bool.pmap_lut3(inputs, sks, params)
  temp_nodes[1629] = outputs[0]
  temp_nodes[1682] = outputs[1]
  temp_nodes[1696] = outputs[2]
  temp_nodes[1700] = outputs[3]
  inputs = [
      (my_string[106], my_string[105], jaxite_bool.constant(False, params), 1),
      (temp_nodes[1718], temp_nodes[1677], temp_nodes[1672], 64),
      (
          temp_nodes[1630],
          temp_nodes[1718],
          jaxite_bool.constant(False, params),
          8,
      ),
      (temp_nodes[1632], temp_nodes[1754], temp_nodes[1748], 7),
  ]
  outputs = jaxite_bool.pmap_lut3(inputs, sks, params)
  temp_nodes[1701] = outputs[0]
  temp_nodes[1717] = outputs[1]
  temp_nodes[1752] = outputs[2]
  temp_nodes[1753] = outputs[3]
  inputs = [
      (temp_nodes[1754], temp_nodes[1748], temp_nodes[1672], 112),
      (temp_nodes[1755], temp_nodes[1632], temp_nodes[1757], 128),
      (my_string[95], temp_nodes[1691], temp_nodes[1489], 16),
      (temp_nodes[1695], my_string[99], my_string[102], 244),
  ]
  outputs = jaxite_bool.pmap_lut3(inputs, sks, params)
  temp_nodes[1758] = outputs[0]
  temp_nodes[1787] = outputs[1]
  temp_nodes[1490] = outputs[2]
  temp_nodes[1491] = outputs[3]
  inputs = [
      (
          temp_nodes[1629],
          temp_nodes[1682],
          jaxite_bool.constant(False, params),
          4,
      ),
      (temp_nodes[1490], temp_nodes[1691], my_string[95], 1),
      (
          temp_nodes[1700],
          my_string[110],
          jaxite_bool.constant(False, params),
          1,
      ),
      (my_string[113], my_string[114], jaxite_bool.constant(False, params), 1),
  ]
  outputs = jaxite_bool.pmap_lut3(inputs, sks, params)
  temp_nodes[1628] = outputs[0]
  temp_nodes[1687] = outputs[1]
  temp_nodes[1702] = outputs[2]
  temp_nodes[1706] = outputs[3]
  inputs = [
      (
          temp_nodes[1717],
          temp_nodes[1682],
          jaxite_bool.constant(False, params),
          4,
      ),
      (
          temp_nodes[1718],
          temp_nodes[1631],
          jaxite_bool.constant(False, params),
          1,
      ),
      (
          temp_nodes[1717],
          temp_nodes[1631],
          jaxite_bool.constant(False, params),
          8,
      ),
      (temp_nodes[1753], temp_nodes[1758], temp_nodes[1752], 180),
  ]
  outputs = jaxite_bool.pmap_lut3(inputs, sks, params)
  temp_nodes[1716] = outputs[0]
  temp_nodes[1747] = outputs[1]
  temp_nodes[1750] = outputs[2]
  temp_nodes[1751] = outputs[3]
  inputs = [
      (
          temp_nodes[1753],
          temp_nodes[1752],
          jaxite_bool.constant(False, params),
          4,
      ),
      (temp_nodes[1787], temp_nodes[1758], temp_nodes[1672], 176),
      (my_string[103], temp_nodes[1696], temp_nodes[1491], 16),
      (temp_nodes[1701], my_string[107], my_string[110], 244),
  ]
  outputs = jaxite_bool.pmap_lut3(inputs, sks, params)
  temp_nodes[1785] = outputs[0]
  temp_nodes[1786] = outputs[1]
  temp_nodes[1492] = outputs[2]
  temp_nodes[1493] = outputs[3]
  inputs = [
      (
          temp_nodes[1628],
          temp_nodes[1687],
          jaxite_bool.constant(False, params),
          4,
      ),
      (temp_nodes[1492], temp_nodes[1696], my_string[103], 1),
      (my_string[117], my_string[116], my_string[118], 7),
      (my_string[124], my_string[125], jaxite_bool.constant(False, params), 8),
  ]
  outputs = jaxite_bool.pmap_lut3(inputs, sks, params)
  temp_nodes[1627] = outputs[0]
  temp_nodes[1692] = outputs[1]
  temp_nodes[1707] = outputs[2]
  temp_nodes[1709] = outputs[3]
  inputs = [
      (my_string[122], my_string[121], jaxite_bool.constant(False, params), 1),
      (
          temp_nodes[1716],
          temp_nodes[1687],
          jaxite_bool.constant(False, params),
          8,
      ),
      (temp_nodes[1747], temp_nodes[1628], temp_nodes[1677], 128),
      (temp_nodes[1677], temp_nodes[1751], temp_nodes[1750], 7),
  ]
  outputs = jaxite_bool.pmap_lut3(inputs, sks, params)
  temp_nodes[1711] = outputs[0]
  temp_nodes[1715] = outputs[1]
  temp_nodes[1746] = outputs[2]
  temp_nodes[1749] = outputs[3]
  inputs = [
      (temp_nodes[1751], temp_nodes[1750], temp_nodes[1682], 112),
      (temp_nodes[1786], temp_nodes[1785], temp_nodes[1677], 224),
      (
          temp_nodes[1751],
          temp_nodes[1677],
          jaxite_bool.constant(False, params),
          8,
      ),
      (my_string[111], temp_nodes[1702], temp_nodes[1493], 16),
  ]
  outputs = jaxite_bool.pmap_lut3(inputs, sks, params)
  temp_nodes[1759] = outputs[0]
  temp_nodes[1784] = outputs[1]
  temp_nodes[1806] = outputs[2]
  temp_nodes[1494] = outputs[3]
  inputs = [
      (temp_nodes[1706], my_string[115], my_string[118], 244),
  ]
  outputs = jaxite_bool.pmap_lut3(inputs, sks, params)
  temp_nodes[1495] = outputs[0]
  inputs = [
      (
          temp_nodes[1627],
          temp_nodes[1692],
          jaxite_bool.constant(False, params),
          4,
      ),
      (temp_nodes[1494], temp_nodes[1702], my_string[111], 1),
      (
          temp_nodes[1709],
          my_string[126],
          jaxite_bool.constant(False, params),
          1,
      ),
      (
          temp_nodes[1629],
          temp_nodes[1715],
          jaxite_bool.constant(False, params),
          4,
      ),
  ]
  outputs = jaxite_bool.pmap_lut3(inputs, sks, params)
  temp_nodes[1626] = outputs[0]
  temp_nodes[1697] = outputs[1]
  temp_nodes[1712] = outputs[2]
  temp_nodes[1737] = outputs[3]
  inputs = [
      (
          temp_nodes[1716],
          temp_nodes[1627],
          jaxite_bool.constant(False, params),
          4,
      ),
      (
          temp_nodes[1628],
          temp_nodes[1715],
          jaxite_bool.constant(False, params),
          4,
      ),
      (temp_nodes[1749], temp_nodes[1759], temp_nodes[1746], 180),
      (
          temp_nodes[1749],
          temp_nodes[1746],
          jaxite_bool.constant(False, params),
          4,
      ),
  ]
  outputs = jaxite_bool.pmap_lut3(inputs, sks, params)
  temp_nodes[1739] = outputs[0]
  temp_nodes[1744] = outputs[1]
  temp_nodes[1745] = outputs[2]
  temp_nodes[1782] = outputs[3]
  inputs = [
      (temp_nodes[1784], temp_nodes[1759], temp_nodes[1682], 176),
      (temp_nodes[1806], temp_nodes[1750], temp_nodes[1682], 96),
      (
          temp_nodes[1747],
          temp_nodes[1687],
          jaxite_bool.constant(False, params),
          1,
      ),
      (my_string[119], temp_nodes[1707], temp_nodes[1495], 16),
  ]
  outputs = jaxite_bool.pmap_lut3(inputs, sks, params)
  temp_nodes[1783] = outputs[0]
  temp_nodes[1805] = outputs[1]
  temp_nodes[1857] = outputs[2]
  temp_nodes[1496] = outputs[3]
  inputs = [
      (temp_nodes[1711], my_string[123], my_string[126], 244),
  ]
  outputs = jaxite_bool.pmap_lut3(inputs, sks, params)
  temp_nodes[1497] = outputs[0]
  inputs = [
      (
          temp_nodes[1626],
          temp_nodes[1697],
          jaxite_bool.constant(False, params),
          4,
      ),
      (temp_nodes[1496], temp_nodes[1707], my_string[119], 1),
      (
          temp_nodes[1715],
          temp_nodes[1692],
          jaxite_bool.constant(False, params),
          4,
      ),
      (my_string[129], my_string[134], my_string[130], 1),
  ]
  outputs = jaxite_bool.pmap_lut3(inputs, sks, params)
  temp_nodes[1625] = outputs[0]
  temp_nodes[1703] = outputs[1]
  temp_nodes[1714] = outputs[2]
  temp_nodes[1727] = outputs[3]
  inputs = [
      (my_string[133], my_string[132], my_string[134], 7),
      (
          temp_nodes[1737],
          temp_nodes[1692],
          jaxite_bool.constant(False, params),
          8,
      ),
      (temp_nodes[1692], temp_nodes[1739], temp_nodes[1697], 112),
      (temp_nodes[1745], temp_nodes[1687], temp_nodes[1744], 120),
  ]
  outputs = jaxite_bool.pmap_lut3(inputs, sks, params)
  temp_nodes[1730] = outputs[0]
  temp_nodes[1736] = outputs[1]
  temp_nodes[1738] = outputs[2]
  temp_nodes[1743] = outputs[3]
  inputs = [
      (
          temp_nodes[1697],
          temp_nodes[1692],
          jaxite_bool.constant(False, params),
          8,
      ),
      (temp_nodes[1783], temp_nodes[1782], temp_nodes[1687], 224),
      (temp_nodes[1805], temp_nodes[1746], temp_nodes[1687], 96),
      (
          temp_nodes[1857],
          temp_nodes[1628],
          jaxite_bool.constant(False, params),
          8,
      ),
  ]
  outputs = jaxite_bool.pmap_lut3(inputs, sks, params)
  temp_nodes[1761] = outputs[0]
  temp_nodes[1781] = outputs[1]
  temp_nodes[1804] = outputs[2]
  temp_nodes[1856] = outputs[3]
  inputs = [
      (temp_nodes[1717], temp_nodes[1682], temp_nodes[1629], 16),
      (
          temp_nodes[1805],
          temp_nodes[1746],
          jaxite_bool.constant(False, params),
          8,
      ),
      (temp_nodes[1784], temp_nodes[1759], temp_nodes[1682], 176),
      (temp_nodes[1686], my_string[87], jaxite_bool.constant(False, params), 8),
  ]
  outputs = jaxite_bool.pmap_lut3(inputs, sks, params)
  temp_nodes[1866] = outputs[0]
  temp_nodes[1879] = outputs[1]
  temp_nodes[1880] = outputs[2]
  temp_nodes[1915] = outputs[3]
  inputs = [
      (my_string[23], my_string[15], jaxite_bool.constant(False, params), 1),
      (temp_nodes[1686], my_string[87], jaxite_bool.constant(False, params), 4),
      (temp_nodes[1635], my_string[63], jaxite_bool.constant(False, params), 4),
      (temp_nodes[1635], my_string[63], jaxite_bool.constant(False, params), 8),
  ]
  outputs = jaxite_bool.pmap_lut3(inputs, sks, params)
  temp_nodes[1920] = outputs[0]
  temp_nodes[1928] = outputs[1]
  temp_nodes[1974] = outputs[2]
  temp_nodes[1976] = outputs[3]
  inputs = [
      (my_string[127], temp_nodes[1712], temp_nodes[1497], 16),
  ]
  outputs = jaxite_bool.pmap_lut3(inputs, sks, params)
  temp_nodes[1498] = outputs[0]
  inputs = [
      (
          temp_nodes[1625],
          temp_nodes[1703],
          jaxite_bool.constant(False, params),
          4,
      ),
      (temp_nodes[1498], temp_nodes[1712], my_string[127], 1),
      (temp_nodes[1727], temp_nodes[1730], my_string[135], 1),
      (
          temp_nodes[1736],
          temp_nodes[1738],
          jaxite_bool.constant(False, params),
          4,
      ),
  ]
  outputs = jaxite_bool.pmap_lut3(inputs, sks, params)
  temp_nodes[1624] = outputs[0]
  temp_nodes[1708] = outputs[1]
  temp_nodes[1726] = outputs[2]
  temp_nodes[1735] = outputs[3]
  inputs = [
      (
          temp_nodes[1625],
          temp_nodes[1714],
          jaxite_bool.constant(False, params),
          8,
      ),
      (
          temp_nodes[1743],
          temp_nodes[1737],
          jaxite_bool.constant(False, params),
          1,
      ),
      (temp_nodes[1743], temp_nodes[1736], temp_nodes[1761], 112),
      (my_string[140], my_string[141], jaxite_bool.constant(False, params), 8),
  ]
  outputs = jaxite_bool.pmap_lut3(inputs, sks, params)
  temp_nodes[1741] = outputs[0]
  temp_nodes[1742] = outputs[1]
  temp_nodes[1760] = outputs[2]
  temp_nodes[1765] = outputs[3]
  inputs = [
      (my_string[137], my_string[138], jaxite_bool.constant(False, params), 1),
      (temp_nodes[1736], temp_nodes[1743], temp_nodes[1697], 128),
      (temp_nodes[1744], temp_nodes[1672], temp_nodes[1781], 7),
      (
          temp_nodes[1804],
          temp_nodes[1736],
          jaxite_bool.constant(False, params),
          8,
      ),
  ]
  outputs = jaxite_bool.pmap_lut3(inputs, sks, params)
  temp_nodes[1767] = outputs[0]
  temp_nodes[1779] = outputs[1]
  temp_nodes[1780] = outputs[2]
  temp_nodes[1803] = outputs[3]
  inputs = [
      (
          temp_nodes[1784],
          temp_nodes[1866],
          jaxite_bool.constant(False, params),
          8,
      ),
      (temp_nodes[1880], temp_nodes[1879], temp_nodes[1687], 224),
      (
          temp_nodes[1880],
          temp_nodes[1856],
          jaxite_bool.constant(False, params),
          8,
      ),
      (temp_nodes[1676], my_string[71], jaxite_bool.constant(False, params), 8),
  ]
  outputs = jaxite_bool.pmap_lut3(inputs, sks, params)
  temp_nodes[1865] = outputs[0]
  temp_nodes[1878] = outputs[1]
  temp_nodes[1889] = outputs[2]
  temp_nodes[1913] = outputs[3]
  inputs = [
      (my_string[38], my_string[39], jaxite_bool.constant(False, params), 1),
      (my_string[79], temp_nodes[1681], temp_nodes[1928], 13),
      (
          temp_nodes[1915],
          temp_nodes[1974],
          jaxite_bool.constant(False, params),
          1,
      ),
      (my_string[79], temp_nodes[1681], temp_nodes[1976], 7),
  ]
  outputs = jaxite_bool.pmap_lut3(inputs, sks, params)
  temp_nodes[1919] = outputs[0]
  temp_nodes[1927] = outputs[1]
  temp_nodes[1973] = outputs[2]
  temp_nodes[1975] = outputs[3]
  inputs = [
      (temp_nodes[1920], my_string[27], my_string[29], 128),
      (my_string[47], my_string[39], temp_nodes[1648], 16),
  ]
  outputs = jaxite_bool.pmap_lut3(inputs, sks, params)
  temp_nodes[41] = outputs[0]
  temp_nodes[43] = outputs[1]
  inputs = [
      (
          temp_nodes[1624],
          temp_nodes[1708],
          jaxite_bool.constant(False, params),
          4,
      ),
      (temp_nodes[1697], temp_nodes[1714], temp_nodes[1703], 112),
      (my_string[131], my_string[134], temp_nodes[1726], 224),
      (
          temp_nodes[1624],
          temp_nodes[1735],
          jaxite_bool.constant(False, params),
          8,
      ),
  ]
  outputs = jaxite_bool.pmap_lut3(inputs, sks, params)
  temp_nodes[1623] = outputs[0]
  temp_nodes[1713] = outputs[1]
  temp_nodes[1725] = outputs[2]
  temp_nodes[1734] = outputs[3]
  inputs = [
      (temp_nodes[1742], temp_nodes[1760], temp_nodes[1741], 180),
      (
          temp_nodes[1765],
          my_string[142],
          jaxite_bool.constant(False, params),
          1,
      ),
      (temp_nodes[1761], temp_nodes[1780], temp_nodes[1779], 13),
      (my_string[146], my_string[145], jaxite_bool.constant(False, params), 1),
  ]
  outputs = jaxite_bool.pmap_lut3(inputs, sks, params)
  temp_nodes[1740] = outputs[0]
  temp_nodes[1768] = outputs[1]
  temp_nodes[1778] = outputs[2]
  temp_nodes[1794] = outputs[3]
  inputs = [
      (temp_nodes[1742], temp_nodes[1803], temp_nodes[1761], 16),
      (
          temp_nodes[1697],
          temp_nodes[1626],
          jaxite_bool.constant(False, params),
          4,
      ),
      (temp_nodes[1745], temp_nodes[1692], temp_nodes[1739], 16),
      (temp_nodes[1677], temp_nodes[1747], temp_nodes[1786], 64),
  ]
  outputs = jaxite_bool.pmap_lut3(inputs, sks, params)
  temp_nodes[1802] = outputs[0]
  temp_nodes[1824] = outputs[1]
  temp_nodes[1825] = outputs[2]
  temp_nodes[1867] = outputs[3]
  inputs = [
      (temp_nodes[1744], temp_nodes[1804], temp_nodes[1878], 7),
      (
          temp_nodes[1889],
          temp_nodes[1865],
          jaxite_bool.constant(False, params),
          1,
      ),
      (my_string[79], temp_nodes[1681], temp_nodes[1913], 7),
      (my_string[95], temp_nodes[1691], temp_nodes[1915], 7),
  ]
  outputs = jaxite_bool.pmap_lut3(inputs, sks, params)
  temp_nodes[1877] = outputs[0]
  temp_nodes[1888] = outputs[1]
  temp_nodes[1912] = outputs[2]
  temp_nodes[1914] = outputs[3]
  inputs = [
      (my_string[33], my_string[34], temp_nodes[1919], 64),
      (my_string[55], my_string[47], jaxite_bool.constant(False, params), 1),
      (temp_nodes[1691], my_string[95], jaxite_bool.constant(False, params), 4),
      (my_string[31], my_string[35], jaxite_bool.constant(False, params), 4),
  ]
  outputs = jaxite_bool.pmap_lut3(inputs, sks, params)
  temp_nodes[1918] = outputs[0]
  temp_nodes[1921] = outputs[1]
  temp_nodes[1922] = outputs[2]
  temp_nodes[1924] = outputs[3]
  inputs = [
      (my_string[15], temp_nodes[1644], temp_nodes[1927], 224),
      (temp_nodes[1676], my_string[71], jaxite_bool.constant(False, params), 4),
      (temp_nodes[1658], my_string[47], jaxite_bool.constant(False, params), 1),
      (temp_nodes[1664], my_string[39], jaxite_bool.constant(False, params), 1),
  ]
  outputs = jaxite_bool.pmap_lut3(inputs, sks, params)
  temp_nodes[1926] = outputs[0]
  temp_nodes[1929] = outputs[1]
  temp_nodes[1930] = outputs[2]
  temp_nodes[1971] = outputs[3]
  inputs = [
      (temp_nodes[1920], my_string[35], my_string[37], 128),
      (my_string[31], my_string[34], temp_nodes[1919], 64),
      (temp_nodes[1913], temp_nodes[1973], temp_nodes[41], 64),
      (
          temp_nodes[43],
          temp_nodes[1975],
          jaxite_bool.constant(False, params),
          8,
      ),
  ]
  outputs = jaxite_bool.pmap_lut3(inputs, sks, params)
  temp_nodes[26] = outputs[0]
  temp_nodes[27] = outputs[1]
  temp_nodes[40] = outputs[2]
  temp_nodes[42] = outputs[3]
  inputs = [
      (my_string[7], my_string[26], temp_nodes[1655], 64),
      (temp_nodes[1767], my_string[139], my_string[142], 244),
  ]
  outputs = jaxite_bool.pmap_lut3(inputs, sks, params)
  temp_nodes[44] = outputs[0]
  temp_nodes[1499] = outputs[1]
  inputs = [
      (
          temp_nodes[1713],
          temp_nodes[1623],
          jaxite_bool.constant(False, params),
          4,
      ),
      (
          temp_nodes[1713],
          temp_nodes[1708],
          jaxite_bool.constant(False, params),
          8,
      ),
      (temp_nodes[1725], temp_nodes[1730], my_string[135], 1),
      (temp_nodes[1703], temp_nodes[1740], temp_nodes[1734], 7),
  ]
  outputs = jaxite_bool.pmap_lut3(inputs, sks, params)
  temp_nodes[1622] = outputs[0]
  temp_nodes[1723] = outputs[1]
  temp_nodes[1724] = outputs[2]
  temp_nodes[1733] = outputs[3]
  inputs = [
      (
          temp_nodes[1734],
          temp_nodes[1740],
          jaxite_bool.constant(False, params),
          8,
      ),
      (temp_nodes[1741], temp_nodes[1682], temp_nodes[1778], 112),
      (my_string[149], my_string[148], my_string[150], 7),
      (temp_nodes[1802], temp_nodes[1741], temp_nodes[1703], 96),
  ]
  outputs = jaxite_bool.pmap_lut3(inputs, sks, params)
  temp_nodes[1762] = outputs[0]
  temp_nodes[1777] = outputs[1]
  temp_nodes[1795] = outputs[2]
  temp_nodes[1801] = outputs[3]
  inputs = [
      (my_string[153], my_string[154], jaxite_bool.constant(False, params), 1),
      (
          temp_nodes[1802],
          temp_nodes[1741],
          jaxite_bool.constant(False, params),
          6,
      ),
      (temp_nodes[1735], temp_nodes[1708], temp_nodes[1624], 16),
      (temp_nodes[1714], temp_nodes[1703], temp_nodes[1625], 16),
  ]
  outputs = jaxite_bool.pmap_lut3(inputs, sks, params)
  temp_nodes[1812] = outputs[0]
  temp_nodes[1820] = outputs[1]
  temp_nodes[1821] = outputs[2]
  temp_nodes[1850] = outputs[3]
  inputs = [
      (temp_nodes[1692], temp_nodes[1877], temp_nodes[1803], 13),
      (temp_nodes[1877], temp_nodes[1742], temp_nodes[1824], 64),
      (
          temp_nodes[1878],
          temp_nodes[1825],
          jaxite_bool.constant(False, params),
          8,
      ),
      (
          temp_nodes[1867],
          temp_nodes[1888],
          jaxite_bool.constant(False, params),
          4,
      ),
  ]
  outputs = jaxite_bool.pmap_lut3(inputs, sks, params)
  temp_nodes[1876] = outputs[0]
  temp_nodes[1886] = outputs[1]
  temp_nodes[1887] = outputs[2]
  temp_nodes[1900] = outputs[3]
  inputs = [
      (
          temp_nodes[1865],
          temp_nodes[1867],
          jaxite_bool.constant(False, params),
          1,
      ),
      (
          temp_nodes[1486],
          temp_nodes[1488],
          jaxite_bool.constant(False, params),
          1,
      ),
      (
          temp_nodes[1912],
          temp_nodes[1914],
          jaxite_bool.constant(False, params),
          8,
      ),
      (temp_nodes[1918], temp_nodes[1920], temp_nodes[1921], 128),
  ]
  outputs = jaxite_bool.pmap_lut3(inputs, sks, params)
  temp_nodes[1905] = outputs[0]
  temp_nodes[1908] = outputs[1]
  temp_nodes[1911] = outputs[2]
  temp_nodes[1917] = outputs[3]
  inputs = [
      (temp_nodes[1924], my_string[32], my_string[37], 128),
      (
          temp_nodes[1929],
          temp_nodes[1926],
          jaxite_bool.constant(False, params),
          4,
      ),
      (my_string[21], my_string[20], my_string[22], 7),
      (temp_nodes[1656], my_string[28], my_string[29], 64),
  ]
  outputs = jaxite_bool.pmap_lut3(inputs, sks, params)
  temp_nodes[1923] = outputs[0]
  temp_nodes[1925] = outputs[1]
  temp_nodes[1933] = outputs[2]
  temp_nodes[1935] = outputs[3]
  inputs = [
      (
          temp_nodes[1930],
          temp_nodes[1971],
          jaxite_bool.constant(False, params),
          1,
      ),
      (temp_nodes[1922], temp_nodes[26], temp_nodes[27], 64),
      (temp_nodes[40], temp_nodes[42], temp_nodes[44], 128),
      (my_string[143], temp_nodes[1768], temp_nodes[1499], 16),
  ]
  outputs = jaxite_bool.pmap_lut3(inputs, sks, params)
  temp_nodes[1970] = outputs[0]
  temp_nodes[25] = outputs[1]
  temp_nodes[39] = outputs[2]
  temp_nodes[1500] = outputs[3]
  inputs = [
      (temp_nodes[1794], my_string[147], my_string[150], 244),
  ]
  outputs = jaxite_bool.pmap_lut3(inputs, sks, params)
  temp_nodes[1501] = outputs[0]
  inputs = [
      (temp_nodes[1625], temp_nodes[1723], temp_nodes[1622], 11),
      (temp_nodes[1733], temp_nodes[1762], temp_nodes[1708], 16),
      (
          temp_nodes[1623],
          temp_nodes[1713],
          jaxite_bool.constant(False, params),
          8,
      ),
      (temp_nodes[1500], temp_nodes[1768], my_string[143], 1),
  ]
  outputs = jaxite_bool.pmap_lut3(inputs, sks, params)
  temp_nodes[1621] = outputs[0]
  temp_nodes[1732] = outputs[1]
  temp_nodes[1763] = outputs[2]
  temp_nodes[1764] = outputs[3]
  inputs = [
      (
          temp_nodes[1623],
          temp_nodes[1724],
          jaxite_bool.constant(False, params),
          4,
      ),
      (temp_nodes[1703], temp_nodes[1777], temp_nodes[1762], 13),
      (temp_nodes[1801], temp_nodes[1734], temp_nodes[1708], 96),
      (my_string[157], my_string[156], my_string[158], 7),
  ]
  outputs = jaxite_bool.pmap_lut3(inputs, sks, params)
  temp_nodes[1771] = outputs[0]
  temp_nodes[1788] = outputs[1]
  temp_nodes[1800] = outputs[2]
  temp_nodes[1813] = outputs[3]
  inputs = [
      (
          temp_nodes[1820],
          temp_nodes[1821],
          jaxite_bool.constant(False, params),
          4,
      ),
      (temp_nodes[1780], temp_nodes[1742], temp_nodes[1824], 64),
      (
          temp_nodes[1802],
          temp_nodes[1850],
          jaxite_bool.constant(False, params),
          4,
      ),
      (
          temp_nodes[1876],
          temp_nodes[1697],
          jaxite_bool.constant(False, params),
          4,
      ),
  ]
  outputs = jaxite_bool.pmap_lut3(inputs, sks, params)
  temp_nodes[1819] = outputs[0]
  temp_nodes[1823] = outputs[1]
  temp_nodes[1849] = outputs[2]
  temp_nodes[1875] = outputs[3]
  inputs = [
      (
          temp_nodes[1802],
          temp_nodes[1741],
          jaxite_bool.constant(False, params),
          8,
      ),
      (
          temp_nodes[1886],
          temp_nodes[1887],
          jaxite_bool.constant(False, params),
          1,
      ),
      (temp_nodes[1900], my_string[20], my_string[21], 128),
      (temp_nodes[1647], temp_nodes[1905], my_string[14], 7),
  ]
  outputs = jaxite_bool.pmap_lut3(inputs, sks, params)
  temp_nodes[1881] = outputs[0]
  temp_nodes[1885] = outputs[1]
  temp_nodes[1903] = outputs[2]
  temp_nodes[1904] = outputs[3]
  inputs = [
      (temp_nodes[1922], temp_nodes[1917], temp_nodes[1923], 64),
      (temp_nodes[1639], temp_nodes[1933], my_string[23], 1),
      (my_string[30], temp_nodes[1935], my_string[31], 14),
      (
          temp_nodes[1484],
          temp_nodes[1908],
          jaxite_bool.constant(False, params),
          4,
      ),
  ]
  outputs = jaxite_bool.pmap_lut3(inputs, sks, params)
  temp_nodes[1916] = outputs[0]
  temp_nodes[1932] = outputs[1]
  temp_nodes[1934] = outputs[2]
  temp_nodes[22] = outputs[3]
  inputs = [
      (temp_nodes[1911], temp_nodes[25], temp_nodes[1921], 128),
      (temp_nodes[1925], temp_nodes[1970], temp_nodes[39], 128),
      (temp_nodes[1480], temp_nodes[1925], my_string[29], 64),
      (my_string[151], temp_nodes[1795], temp_nodes[1501], 16),
  ]
  outputs = jaxite_bool.pmap_lut3(inputs, sks, params)
  temp_nodes[24] = outputs[0]
  temp_nodes[38] = outputs[1]
  temp_nodes[334] = outputs[2]
  temp_nodes[1502] = outputs[3]
  inputs = [
      (temp_nodes[1812], my_string[155], my_string[158], 244),
  ]
  outputs = jaxite_bool.pmap_lut3(inputs, sks, params)
  temp_nodes[1503] = outputs[0]
  inputs = [
      (temp_nodes[1621], temp_nodes[1623], temp_nodes[1724], 16),
      (temp_nodes[1732], temp_nodes[1763], temp_nodes[1724], 96),
      (
          temp_nodes[1771],
          temp_nodes[1764],
          jaxite_bool.constant(False, params),
          4,
      ),
      (
          temp_nodes[1723],
          temp_nodes[1724],
          jaxite_bool.constant(False, params),
          4,
      ),
  ]
  outputs = jaxite_bool.pmap_lut3(inputs, sks, params)
  temp_nodes[1620] = outputs[0]
  temp_nodes[1731] = outputs[1]
  temp_nodes[1770] = outputs[2]
  temp_nodes[1772] = outputs[3]
  inputs = [
      (
          temp_nodes[1732],
          temp_nodes[1763],
          jaxite_bool.constant(False, params),
          8,
      ),
      (temp_nodes[1788], temp_nodes[1787], temp_nodes[1708], 16),
      (temp_nodes[1502], temp_nodes[1795], my_string[151], 1),
      (temp_nodes[1800], temp_nodes[1763], temp_nodes[1724], 96),
  ]
  outputs = jaxite_bool.pmap_lut3(inputs, sks, params)
  temp_nodes[1775] = outputs[0]
  temp_nodes[1776] = outputs[1]
  temp_nodes[1791] = outputs[2]
  temp_nodes[1799] = outputs[3]
  inputs = [
      (temp_nodes[1724], temp_nodes[1621], temp_nodes[1764], 208),
      (temp_nodes[1764], temp_nodes[1771], temp_nodes[1621], 64),
      (
          temp_nodes[1777],
          temp_nodes[1819],
          jaxite_bool.constant(False, params),
          4,
      ),
      (temp_nodes[1825], temp_nodes[1781], temp_nodes[1823], 7),
  ]
  outputs = jaxite_bool.pmap_lut3(inputs, sks, params)
  temp_nodes[1808] = outputs[0]
  temp_nodes[1816] = outputs[1]
  temp_nodes[1818] = outputs[2]
  temp_nodes[1822] = outputs[3]
  inputs = [
      (temp_nodes[1800], temp_nodes[1724], temp_nodes[1622], 16),
      (my_string[162], my_string[161], jaxite_bool.constant(False, params), 1),
      (
          temp_nodes[1778],
          temp_nodes[1849],
          jaxite_bool.constant(False, params),
          4,
      ),
      (
          temp_nodes[1783],
          temp_nodes[1856],
          jaxite_bool.constant(False, params),
          8,
      ),
  ]
  outputs = jaxite_bool.pmap_lut3(inputs, sks, params)
  temp_nodes[1827] = outputs[0]
  temp_nodes[1842] = outputs[1]
  temp_nodes[1848] = outputs[2]
  temp_nodes[1855] = outputs[3]
  inputs = [
      (temp_nodes[1881], temp_nodes[1875], temp_nodes[1703], 224),
      (
          temp_nodes[1885],
          temp_nodes[1888],
          jaxite_bool.constant(False, params),
          8,
      ),
      (my_string[22], temp_nodes[1903], temp_nodes[1904], 14),
      (temp_nodes[1484], temp_nodes[1911], temp_nodes[1916], 64),
  ]
  outputs = jaxite_bool.pmap_lut3(inputs, sks, params)
  temp_nodes[1874] = outputs[0]
  temp_nodes[1884] = outputs[1]
  temp_nodes[1902] = outputs[2]
  temp_nodes[1910] = outputs[3]
  inputs = [
      (
          temp_nodes[1932],
          temp_nodes[1934],
          jaxite_bool.constant(False, params),
          1,
      ),
      (
          temp_nodes[1820],
          temp_nodes[1734],
          jaxite_bool.constant(False, params),
          8,
      ),
      (
          temp_nodes[1875],
          temp_nodes[1881],
          jaxite_bool.constant(False, params),
          1,
      ),
      (temp_nodes[1849], temp_nodes[1875], temp_nodes[1867], 7),
  ]
  outputs = jaxite_bool.pmap_lut3(inputs, sks, params)
  temp_nodes[1931] = outputs[0]
  temp_nodes[1940] = outputs[1]
  temp_nodes[1942] = outputs[2]
  temp_nodes[1945] = outputs[3]
  inputs = [
      (temp_nodes[1652], temp_nodes[1867], my_string[6], 13),
      (my_string[7], my_string[29], jaxite_bool.constant(False, params), 4),
      (temp_nodes[1490], temp_nodes[1925], temp_nodes[24], 64),
      (temp_nodes[1480], temp_nodes[38], temp_nodes[22], 64),
  ]
  outputs = jaxite_bool.pmap_lut3(inputs, sks, params)
  temp_nodes[1967] = outputs[0]
  temp_nodes[1] = outputs[1]
  temp_nodes[23] = outputs[2]
  temp_nodes[37] = outputs[3]
  inputs = [
      (temp_nodes[1932], temp_nodes[334], temp_nodes[39], 64),
      (my_string[159], temp_nodes[1813], temp_nodes[1503], 16),
  ]
  outputs = jaxite_bool.pmap_lut3(inputs, sks, params)
  temp_nodes[332] = outputs[0]
  temp_nodes[1504] = outputs[1]
  inputs = [
      (temp_nodes[1620], temp_nodes[1731], temp_nodes[1764], 96),
      (my_string[54], my_string[55], my_string[51], 16),
      (
          temp_nodes[1770],
          temp_nodes[1772],
          jaxite_bool.constant(False, params),
          8,
      ),
      (
          temp_nodes[1620],
          temp_nodes[1731],
          jaxite_bool.constant(False, params),
          8,
      ),
  ]
  outputs = jaxite_bool.pmap_lut3(inputs, sks, params)
  temp_nodes[1619] = outputs[0]
  temp_nodes[1669] = outputs[1]
  temp_nodes[1769] = outputs[2]
  temp_nodes[1773] = outputs[3]
  inputs = [
      (temp_nodes[1775], temp_nodes[1776], temp_nodes[1724], 96),
      (temp_nodes[1620], temp_nodes[1799], temp_nodes[1764], 96),
      (temp_nodes[1504], temp_nodes[1813], my_string[159], 1),
      (
          temp_nodes[1799],
          temp_nodes[1816],
          jaxite_bool.constant(False, params),
          4,
      ),
  ]
  outputs = jaxite_bool.pmap_lut3(inputs, sks, params)
  temp_nodes[1774] = outputs[0]
  temp_nodes[1798] = outputs[1]
  temp_nodes[1809] = outputs[2]
  temp_nodes[1815] = outputs[3]
  inputs = [
      (
          temp_nodes[1818],
          temp_nodes[1822],
          jaxite_bool.constant(False, params),
          4,
      ),
      (
          temp_nodes[1776],
          temp_nodes[1827],
          jaxite_bool.constant(False, params),
          8,
      ),
      (
          temp_nodes[1808],
          temp_nodes[1791],
          jaxite_bool.constant(False, params),
          8,
      ),
      (my_string[165], my_string[164], my_string[166], 7),
  ]
  outputs = jaxite_bool.pmap_lut3(inputs, sks, params)
  temp_nodes[1817] = outputs[0]
  temp_nodes[1826] = outputs[1]
  temp_nodes[1833] = outputs[2]
  temp_nodes[1843] = outputs[3]
  inputs = [
      (temp_nodes[1772], temp_nodes[1791], temp_nodes[1770], 16),
      (
          temp_nodes[1818],
          temp_nodes[1848],
          jaxite_bool.constant(False, params),
          1,
      ),
      (temp_nodes[1855], temp_nodes[1865], temp_nodes[1822], 16),
      (
          temp_nodes[1874],
          temp_nodes[1819],
          jaxite_bool.constant(False, params),
          8,
      ),
  ]
  outputs = jaxite_bool.pmap_lut3(inputs, sks, params)
  temp_nodes[1847] = outputs[0]
  temp_nodes[1863] = outputs[1]
  temp_nodes[1864] = outputs[2]
  temp_nodes[1873] = outputs[3]
  inputs = [
      (
          temp_nodes[1875],
          temp_nodes[1849],
          jaxite_bool.constant(False, params),
          8,
      ),
      (
          temp_nodes[1867],
          temp_nodes[1884],
          jaxite_bool.constant(False, params),
          4,
      ),
      (
          temp_nodes[1887],
          temp_nodes[1900],
          jaxite_bool.constant(False, params),
          4,
      ),
      (temp_nodes[1490], temp_nodes[1482], temp_nodes[1908], 16),
  ]
  outputs = jaxite_bool.pmap_lut3(inputs, sks, params)
  temp_nodes[1882] = outputs[0]
  temp_nodes[1894] = outputs[1]
  temp_nodes[1899] = outputs[2]
  temp_nodes[1907] = outputs[3]
  inputs = [
      (temp_nodes[1930], temp_nodes[1925], temp_nodes[1910], 64),
      (temp_nodes[1940], temp_nodes[1874], temp_nodes[1708], 224),
      (
          temp_nodes[1942],
          temp_nodes[1940],
          jaxite_bool.constant(False, params),
          4,
      ),
      (
          temp_nodes[1884],
          temp_nodes[1945],
          jaxite_bool.constant(False, params),
          8,
      ),
  ]
  outputs = jaxite_bool.pmap_lut3(inputs, sks, params)
  temp_nodes[1909] = outputs[0]
  temp_nodes[1939] = outputs[1]
  temp_nodes[1941] = outputs[2]
  temp_nodes[1944] = outputs[3]
  inputs = [
      (
          temp_nodes[1848],
          temp_nodes[1867],
          jaxite_bool.constant(False, params),
          1,
      ),
      (temp_nodes[1888], my_string[28], jaxite_bool.constant(False, params), 8),
      (my_string[25], temp_nodes[1], my_string[24], 64),
      (temp_nodes[1482], temp_nodes[1930], temp_nodes[1902], 16),
  ]
  outputs = jaxite_bool.pmap_lut3(inputs, sks, params)
  temp_nodes[1949] = outputs[0]
  temp_nodes[1966] = outputs[1]
  temp_nodes[0] = outputs[2]
  temp_nodes[20] = outputs[3]
  inputs = [
      (temp_nodes[22], temp_nodes[23], temp_nodes[1931], 128),
      (temp_nodes[1967], temp_nodes[1932], temp_nodes[37], 16),
      (my_string[170], my_string[169], jaxite_bool.constant(False, params), 1),
      (temp_nodes[1667], my_string[39], jaxite_bool.constant(False, params), 4),
  ]
  outputs = jaxite_bool.pmap_lut3(inputs, sks, params)
  temp_nodes[21] = outputs[0]
  temp_nodes[36] = outputs[1]
  temp_nodes[64] = outputs[2]
  temp_nodes[130] = outputs[3]
  inputs = [
      (temp_nodes[1671], my_string[55], jaxite_bool.constant(False, params), 8),
      (temp_nodes[22], temp_nodes[332], temp_nodes[1970], 128),
      (temp_nodes[1842], my_string[163], my_string[166], 244),
  ]
  outputs = jaxite_bool.pmap_lut3(inputs, sks, params)
  temp_nodes[308] = outputs[0]
  temp_nodes[331] = outputs[1]
  temp_nodes[1505] = outputs[2]
  inputs = [
      (
          temp_nodes[1619],
          temp_nodes[1769],
          jaxite_bool.constant(False, params),
          8,
      ),
      (my_string[62], my_string[63], my_string[59], 16),
      (temp_nodes[1773], temp_nodes[1774], temp_nodes[1764], 96),
      (temp_nodes[1798], temp_nodes[1769], temp_nodes[1791], 96),
  ]
  outputs = jaxite_bool.pmap_lut3(inputs, sks, params)
  temp_nodes[1618] = outputs[0]
  temp_nodes[1633] = outputs[1]
  temp_nodes[1790] = outputs[2]
  temp_nodes[1797] = outputs[3]
  inputs = [
      (temp_nodes[1808], temp_nodes[1770], temp_nodes[1791], 16),
      (temp_nodes[1815], temp_nodes[1774], temp_nodes[1817], 112),
      (
          temp_nodes[1770],
          temp_nodes[1833],
          jaxite_bool.constant(False, params),
          4,
      ),
      (temp_nodes[1619], temp_nodes[1769], temp_nodes[1791], 96),
  ]
  outputs = jaxite_bool.pmap_lut3(inputs, sks, params)
  temp_nodes[1807] = outputs[0]
  temp_nodes[1814] = outputs[1]
  temp_nodes[1832] = outputs[2]
  temp_nodes[1834] = outputs[3]
  inputs = [
      (
          temp_nodes[1771],
          temp_nodes[1833],
          jaxite_bool.constant(False, params),
          4,
      ),
      (temp_nodes[1791], temp_nodes[1770], temp_nodes[1809], 208),
      (temp_nodes[1620], temp_nodes[1799], temp_nodes[1847], 144),
      (temp_nodes[1867], temp_nodes[1864], temp_nodes[1863], 64),
  ]
  outputs = jaxite_bool.pmap_lut3(inputs, sks, params)
  temp_nodes[1837] = outputs[0]
  temp_nodes[1838] = outputs[1]
  temp_nodes[1846] = outputs[2]
  temp_nodes[1862] = outputs[3]
  inputs = [
      (
          temp_nodes[1873],
          temp_nodes[1882],
          jaxite_bool.constant(False, params),
          1,
      ),
      (temp_nodes[1884], my_string[52], jaxite_bool.constant(False, params), 8),
      (temp_nodes[1882], temp_nodes[1894], my_string[44], 64),
      (temp_nodes[1899], my_string[28], my_string[29], 128),
  ]
  outputs = jaxite_bool.pmap_lut3(inputs, sks, params)
  temp_nodes[1872] = outputs[0]
  temp_nodes[1883] = outputs[1]
  temp_nodes[1893] = outputs[2]
  temp_nodes[1898] = outputs[3]
  inputs = [
      (
          temp_nodes[1907],
          temp_nodes[1909],
          jaxite_bool.constant(False, params),
          8,
      ),
      (temp_nodes[1941], temp_nodes[1939], temp_nodes[1827], 64),
      (
          temp_nodes[1873],
          temp_nodes[1944],
          jaxite_bool.constant(False, params),
          4,
      ),
      (temp_nodes[1817], temp_nodes[1949], my_string[56], 128),
  ]
  outputs = jaxite_bool.pmap_lut3(inputs, sks, params)
  temp_nodes[1906] = outputs[0]
  temp_nodes[1938] = outputs[1]
  temp_nodes[1943] = outputs[2]
  temp_nodes[1948] = outputs[3]
  inputs = [
      (my_string[58], my_string[61], jaxite_bool.constant(False, params), 8),
      (temp_nodes[1826], temp_nodes[1855], temp_nodes[1865], 1),
      (temp_nodes[1873], temp_nodes[1944], my_string[48], 64),
      (temp_nodes[1669], my_string[50], my_string[53], 128),
  ]
  outputs = jaxite_bool.pmap_lut3(inputs, sks, params)
  temp_nodes[1950] = outputs[0]
  temp_nodes[1954] = outputs[1]
  temp_nodes[1960] = outputs[2]
  temp_nodes[1961] = outputs[3]
  inputs = [
      (temp_nodes[1894], my_string[36], my_string[37], 128),
      (temp_nodes[331], temp_nodes[0], jaxite_bool.constant(False, params), 8),
      (my_string[33], temp_nodes[21], temp_nodes[20], 64),
      (temp_nodes[1966], temp_nodes[1902], temp_nodes[36], 64),
  ]
  outputs = jaxite_bool.pmap_lut3(inputs, sks, params)
  temp_nodes[1964] = outputs[0]
  temp_nodes[1968] = outputs[1]
  temp_nodes[19] = outputs[2]
  temp_nodes[35] = outputs[3]
  inputs = [
      (my_string[173], my_string[172], my_string[174], 7),
      (
          temp_nodes[1696],
          my_string[103],
          jaxite_bool.constant(False, params),
          4,
      ),
      (my_string[31], my_string[23], my_string[45], 16),
      (
          temp_nodes[1941],
          temp_nodes[1939],
          jaxite_bool.constant(False, params),
          4,
      ),
  ]
  outputs = jaxite_bool.pmap_lut3(inputs, sks, params)
  temp_nodes[65] = outputs[0]
  temp_nodes[82] = outputs[1]
  temp_nodes[85] = outputs[2]
  temp_nodes[193] = outputs[3]
  inputs = [
      (temp_nodes[1624], temp_nodes[1800], temp_nodes[1723], 64),
      (temp_nodes[1667], my_string[39], jaxite_bool.constant(False, params), 8),
      (
          temp_nodes[308],
          temp_nodes[130],
          jaxite_bool.constant(False, params),
          1,
      ),
      (my_string[167], temp_nodes[1843], temp_nodes[1505], 16),
  ]
  outputs = jaxite_bool.pmap_lut3(inputs, sks, params)
  temp_nodes[194] = outputs[0]
  temp_nodes[323] = outputs[1]
  temp_nodes[361] = outputs[2]
  temp_nodes[1506] = outputs[3]
  inputs = [
      (temp_nodes[64], my_string[171], my_string[174], 244),
  ]
  outputs = jaxite_bool.pmap_lut3(inputs, sks, params)
  temp_nodes[1507] = outputs[0]
  inputs = [
      (temp_nodes[1773], temp_nodes[1774], temp_nodes[1618], 96),
      (my_string[46], my_string[47], jaxite_bool.constant(False, params), 1),
      (temp_nodes[1790], temp_nodes[1618], temp_nodes[1791], 224),
      (temp_nodes[1797], temp_nodes[1809], temp_nodes[1807], 16),
  ]
  outputs = jaxite_bool.pmap_lut3(inputs, sks, params)
  temp_nodes[1617] = outputs[0]
  temp_nodes[1660] = outputs[1]
  temp_nodes[1789] = outputs[2]
  temp_nodes[1796] = outputs[3]
  inputs = [
      (temp_nodes[1506], temp_nodes[1843], my_string[167], 1),
      (
          temp_nodes[1832],
          temp_nodes[1797],
          jaxite_bool.constant(False, params),
          6,
      ),
      (temp_nodes[1846], temp_nodes[1790], temp_nodes[1848], 7),
      (
          temp_nodes[1826],
          temp_nodes[1862],
          jaxite_bool.constant(False, params),
          4,
      ),
  ]
  outputs = jaxite_bool.pmap_lut3(inputs, sks, params)
  temp_nodes[1839] = outputs[0]
  temp_nodes[1844] = outputs[1]
  temp_nodes[1845] = outputs[2]
  temp_nodes[1861] = outputs[3]
  inputs = [
      (
          temp_nodes[1774],
          temp_nodes[1815],
          jaxite_bool.constant(False, params),
          8,
      ),
      (temp_nodes[1867], temp_nodes[1883], temp_nodes[1872], 64),
      (my_string[54], my_string[55], jaxite_bool.constant(False, params), 1),
      (temp_nodes[1893], my_string[45], jaxite_bool.constant(False, params), 8),
  ]
  outputs = jaxite_bool.pmap_lut3(inputs, sks, params)
  temp_nodes[1868] = outputs[0]
  temp_nodes[1871] = outputs[1]
  temp_nodes[1890] = outputs[2]
  temp_nodes[1892] = outputs[3]
  inputs = [
      (temp_nodes[1894], my_string[36], jaxite_bool.constant(False, params), 8),
      (
          temp_nodes[1898],
          temp_nodes[1655],
          jaxite_bool.constant(False, params),
          4,
      ),
      (temp_nodes[1902], temp_nodes[1906], temp_nodes[1931], 128),
      (temp_nodes[1938], my_string[57], temp_nodes[1943], 16),
  ]
  outputs = jaxite_bool.pmap_lut3(inputs, sks, params)
  temp_nodes[1896] = outputs[0]
  temp_nodes[1897] = outputs[1]
  temp_nodes[1901] = outputs[2]
  temp_nodes[1937] = outputs[3]
  inputs = [
      (temp_nodes[1938], temp_nodes[1943], my_string[60], 64),
      (temp_nodes[1948], temp_nodes[1633], temp_nodes[1950], 128),
      (
          temp_nodes[1814],
          temp_nodes[1954],
          jaxite_bool.constant(False, params),
          8,
      ),
      (temp_nodes[1883], temp_nodes[1960], temp_nodes[1961], 64),
  ]
  outputs = jaxite_bool.pmap_lut3(inputs, sks, params)
  temp_nodes[1946] = outputs[0]
  temp_nodes[1947] = outputs[1]
  temp_nodes[1953] = outputs[2]
  temp_nodes[1959] = outputs[3]
  inputs = [
      (
          temp_nodes[1964],
          temp_nodes[1919],
          jaxite_bool.constant(False, params),
          4,
      ),
      (temp_nodes[1966], temp_nodes[1967], temp_nodes[1968], 16),
      (my_string[62], my_string[63], jaxite_bool.constant(False, params), 1),
      (temp_nodes[19], temp_nodes[1894], my_string[32], 128),
  ]
  outputs = jaxite_bool.pmap_lut3(inputs, sks, params)
  temp_nodes[1963] = outputs[0]
  temp_nodes[1965] = outputs[1]
  temp_nodes[3] = outputs[2]
  temp_nodes[18] = outputs[3]
  inputs = [
      (my_string[25], temp_nodes[35], jaxite_bool.constant(False, params), 4),
      (
          temp_nodes[1837],
          temp_nodes[1838],
          jaxite_bool.constant(False, params),
          8,
      ),
      (
          temp_nodes[1832],
          temp_nodes[1834],
          jaxite_bool.constant(False, params),
          6,
      ),
      (
          temp_nodes[1928],
          temp_nodes[1922],
          jaxite_bool.constant(False, params),
          1,
      ),
  ]
  outputs = jaxite_bool.pmap_lut3(inputs, sks, params)
  temp_nodes[34] = outputs[0]
  temp_nodes[56] = outputs[1]
  temp_nodes[68] = outputs[2]
  temp_nodes[80] = outputs[3]
  inputs = [
      (my_string[79], temp_nodes[1681], temp_nodes[82], 13),
      (
          temp_nodes[1696],
          my_string[103],
          jaxite_bool.constant(False, params),
          8,
      ),
      (temp_nodes[1807], temp_nodes[1837], temp_nodes[1809], 224),
      (my_string[178], my_string[177], jaxite_bool.constant(False, params), 1),
  ]
  outputs = jaxite_bool.pmap_lut3(inputs, sks, params)
  temp_nodes[81] = outputs[0]
  temp_nodes[84] = outputs[1]
  temp_nodes[114] = outputs[2]
  temp_nodes[118] = outputs[3]
  inputs = [
      (temp_nodes[193], temp_nodes[194], temp_nodes[1724], 96),
      (temp_nodes[1621], temp_nodes[1799], temp_nodes[1771], 64),
      (temp_nodes[1671], my_string[55], jaxite_bool.constant(False, params), 4),
      (temp_nodes[1662], my_string[47], jaxite_bool.constant(False, params), 8),
  ]
  outputs = jaxite_bool.pmap_lut3(inputs, sks, params)
  temp_nodes[192] = outputs[0]
  temp_nodes[195] = outputs[1]
  temp_nodes[304] = outputs[2]
  temp_nodes[309] = outputs[3]
  inputs = [
      (temp_nodes[323], temp_nodes[361], temp_nodes[85], 64),
      (my_string[175], temp_nodes[65], temp_nodes[1507], 16),
  ]
  outputs = jaxite_bool.pmap_lut3(inputs, sks, params)
  temp_nodes[360] = outputs[0]
  temp_nodes[1508] = outputs[1]
  inputs = [
      (temp_nodes[1617], temp_nodes[1789], temp_nodes[1796], 64),
      (my_string[70], my_string[71], jaxite_bool.constant(False, params), 1),
      (my_string[78], my_string[79], jaxite_bool.constant(False, params), 1),
      (
          temp_nodes[1617],
          temp_nodes[1789],
          jaxite_bool.constant(False, params),
          4,
      ),
  ]
  outputs = jaxite_bool.pmap_lut3(inputs, sks, params)
  temp_nodes[1616] = outputs[0]
  temp_nodes[1674] = outputs[1]
  temp_nodes[1679] = outputs[2]
  temp_nodes[1830] = outputs[3]
  inputs = [
      (
          temp_nodes[1832],
          temp_nodes[1834],
          jaxite_bool.constant(False, params),
          8,
      ),
      (temp_nodes[1837], temp_nodes[1839], temp_nodes[1838], 16),
      (temp_nodes[1826], temp_nodes[1855], temp_nodes[1845], 16),
      (temp_nodes[1868], temp_nodes[1861], temp_nodes[1673], 64),
  ]
  outputs = jaxite_bool.pmap_lut3(inputs, sks, params)
  temp_nodes[1831] = outputs[0]
  temp_nodes[1836] = outputs[1]
  temp_nodes[1854] = outputs[2]
  temp_nodes[1860] = outputs[3]
  inputs = [
      (my_string[53], temp_nodes[1871], temp_nodes[1890], 112),
      (
          temp_nodes[1892],
          temp_nodes[1660],
          jaxite_bool.constant(False, params),
          4,
      ),
      (temp_nodes[1896], temp_nodes[1897], temp_nodes[1901], 16),
      (temp_nodes[1946], temp_nodes[1937], temp_nodes[1947], 64),
  ]
  outputs = jaxite_bool.pmap_lut3(inputs, sks, params)
  temp_nodes[1870] = outputs[0]
  temp_nodes[1891] = outputs[1]
  temp_nodes[1895] = outputs[2]
  temp_nodes[1936] = outputs[3]
  inputs = [
      (temp_nodes[1845], temp_nodes[1953], my_string[76], 128),
      (my_string[49], temp_nodes[1943], temp_nodes[1959], 64),
      (temp_nodes[1963], temp_nodes[1902], temp_nodes[1965], 64),
      (my_string[61], temp_nodes[1946], temp_nodes[3], 112),
  ]
  outputs = jaxite_bool.pmap_lut3(inputs, sks, params)
  temp_nodes[1952] = outputs[0]
  temp_nodes[1958] = outputs[1]
  temp_nodes[1962] = outputs[2]
  temp_nodes[2] = outputs[3]
  inputs = [
      (temp_nodes[1896], temp_nodes[1897], temp_nodes[18], 16),
      (temp_nodes[34], my_string[24], jaxite_bool.constant(False, params), 8),
      (
          temp_nodes[1844],
          temp_nodes[56],
          jaxite_bool.constant(False, params),
          8,
      ),
      (temp_nodes[1809], temp_nodes[1844], temp_nodes[56], 7),
  ]
  outputs = jaxite_bool.pmap_lut3(inputs, sks, params)
  temp_nodes[17] = outputs[0]
  temp_nodes[33] = outputs[1]
  temp_nodes[55] = outputs[2]
  temp_nodes[57] = outputs[3]
  inputs = [
      (
          temp_nodes[1833],
          temp_nodes[1809],
          jaxite_bool.constant(False, params),
          4,
      ),
      (
          temp_nodes[1838],
          temp_nodes[1839],
          jaxite_bool.constant(False, params),
          4,
      ),
      (temp_nodes[1508], temp_nodes[65], my_string[175], 1),
      (temp_nodes[68], temp_nodes[56], jaxite_bool.constant(False, params), 8),
  ]
  outputs = jaxite_bool.pmap_lut3(inputs, sks, params)
  temp_nodes[59] = outputs[0]
  temp_nodes[60] = outputs[1]
  temp_nodes[61] = outputs[2]
  temp_nodes[67] = outputs[3]
  inputs = [
      (
          temp_nodes[84],
          temp_nodes[1914],
          jaxite_bool.constant(False, params),
          4,
      ),
      (
          temp_nodes[1480],
          temp_nodes[1482],
          jaxite_bool.constant(False, params),
          1,
      ),
      (
          temp_nodes[114],
          temp_nodes[1839],
          jaxite_bool.constant(False, params),
          4,
      ),
      (my_string[181], my_string[180], my_string[182], 7),
  ]
  outputs = jaxite_bool.pmap_lut3(inputs, sks, params)
  temp_nodes[83] = outputs[0]
  temp_nodes[86] = outputs[1]
  temp_nodes[113] = outputs[2]
  temp_nodes[119] = outputs[3]
  inputs = [
      (temp_nodes[1809], temp_nodes[68], temp_nodes[56], 7),
      (my_string[186], my_string[185], jaxite_bool.constant(False, params), 1),
      (temp_nodes[1929], my_string[39], temp_nodes[1921], 16),
      (
          temp_nodes[1702],
          my_string[111],
          jaxite_bool.constant(False, params),
          8,
      ),
  ]
  outputs = jaxite_bool.pmap_lut3(inputs, sks, params)
  temp_nodes[123] = outputs[0]
  temp_nodes[153] = outputs[1]
  temp_nodes[174] = outputs[2]
  temp_nodes[175] = outputs[3]
  inputs = [
      (my_string[95], temp_nodes[1691], temp_nodes[84], 7),
      (
          temp_nodes[1702],
          my_string[111],
          jaxite_bool.constant(False, params),
          4,
      ),
      (
          temp_nodes[192],
          temp_nodes[195],
          jaxite_bool.constant(False, params),
          6,
      ),
      (
          temp_nodes[1798],
          temp_nodes[1769],
          jaxite_bool.constant(False, params),
          8,
      ),
  ]
  outputs = jaxite_bool.pmap_lut3(inputs, sks, params)
  temp_nodes[177] = outputs[0]
  temp_nodes[179] = outputs[1]
  temp_nodes[191] = outputs[2]
  temp_nodes[196] = outputs[3]
  inputs = [
      (temp_nodes[308], temp_nodes[309], temp_nodes[1920], 16),
      (temp_nodes[80], temp_nodes[81], jaxite_bool.constant(False, params), 8),
      (
          temp_nodes[304],
          temp_nodes[1974],
          jaxite_bool.constant(False, params),
          1,
      ),
      (temp_nodes[360], my_string[42], my_string[43], 128),
  ]
  outputs = jaxite_bool.pmap_lut3(inputs, sks, params)
  temp_nodes[307] = outputs[0]
  temp_nodes[317] = outputs[1]
  temp_nodes[319] = outputs[2]
  temp_nodes[359] = outputs[3]
  inputs = [
      (
          temp_nodes[1941],
          temp_nodes[1708],
          jaxite_bool.constant(False, params),
          8,
      ),
      (temp_nodes[118], my_string[179], my_string[182], 244),
      (temp_nodes[193], temp_nodes[194], temp_nodes[195], 23),
  ]
  outputs = jaxite_bool.pmap_lut3(inputs, sks, params)
  temp_nodes[393] = outputs[0]
  temp_nodes[1509] = outputs[1]
  temp_nodes[1527] = outputs[2]
  inputs = [
      (temp_nodes[1616], temp_nodes[1826], temp_nodes[1814], 16),
      (my_string[86], my_string[87], jaxite_bool.constant(False, params), 1),
      (temp_nodes[1830], temp_nodes[1831], temp_nodes[1809], 96),
      (
          temp_nodes[1844],
          temp_nodes[1836],
          jaxite_bool.constant(False, params),
          4,
      ),
  ]
  outputs = jaxite_bool.pmap_lut3(inputs, sks, params)
  temp_nodes[1615] = outputs[0]
  temp_nodes[1684] = outputs[1]
  temp_nodes[1829] = outputs[2]
  temp_nodes[1835] = outputs[3]
  inputs = [
      (temp_nodes[1616], temp_nodes[1854], temp_nodes[1814], 64),
      (
          temp_nodes[1860],
          temp_nodes[1674],
          jaxite_bool.constant(False, params),
          4,
      ),
      (temp_nodes[1870], temp_nodes[1891], temp_nodes[1895], 16),
      (my_string[77], temp_nodes[1952], temp_nodes[1679], 112),
  ]
  outputs = jaxite_bool.pmap_lut3(inputs, sks, params)
  temp_nodes[1853] = outputs[0]
  temp_nodes[1859] = outputs[1]
  temp_nodes[1869] = outputs[2]
  temp_nodes[1951] = outputs[3]
  inputs = [
      (temp_nodes[1891], temp_nodes[1958], temp_nodes[1962], 64),
      (
          temp_nodes[1484],
          temp_nodes[1480],
          jaxite_bool.constant(False, params),
          1,
      ),
      (
          temp_nodes[1868],
          temp_nodes[1826],
          jaxite_bool.constant(False, params),
          1,
      ),
      (temp_nodes[1870], temp_nodes[1936], temp_nodes[17], 64),
  ]
  outputs = jaxite_bool.pmap_lut3(inputs, sks, params)
  temp_nodes[1957] = outputs[0]
  temp_nodes[1972] = outputs[1]
  temp_nodes[5] = outputs[2]
  temp_nodes[16] = outputs[3]
  inputs = [
      (
          temp_nodes[1963],
          temp_nodes[1958],
          jaxite_bool.constant(False, params),
          4,
      ),
      (temp_nodes[1891], temp_nodes[2], temp_nodes[33], 16),
      (temp_nodes[55], temp_nodes[57], temp_nodes[1839], 16),
      (temp_nodes[59], temp_nodes[61], temp_nodes[60], 16),
  ]
  outputs = jaxite_bool.pmap_lut3(inputs, sks, params)
  temp_nodes[31] = outputs[0]
  temp_nodes[32] = outputs[1]
  temp_nodes[54] = outputs[2]
  temp_nodes[58] = outputs[3]
  inputs = [
      (temp_nodes[1846], temp_nodes[1790], temp_nodes[1616], 7),
      (temp_nodes[1492], temp_nodes[1490], temp_nodes[1488], 1),
      (temp_nodes[59], temp_nodes[60], jaxite_bool.constant(False, params), 8),
      (temp_nodes[123], temp_nodes[67], temp_nodes[1839], 16),
  ]
  outputs = jaxite_bool.pmap_lut3(inputs, sks, params)
  temp_nodes[71] = outputs[0]
  temp_nodes[79] = outputs[1]
  temp_nodes[111] = outputs[2]
  temp_nodes[122] = outputs[3]
  inputs = [
      (
          temp_nodes[1929],
          temp_nodes[1974],
          jaxite_bool.constant(False, params),
          1,
      ),
      (temp_nodes[113], temp_nodes[61], jaxite_bool.constant(False, params), 8),
      (my_string[189], my_string[188], my_string[190], 7),
      (temp_nodes[175], temp_nodes[81], temp_nodes[174], 64),
  ]
  outputs = jaxite_bool.pmap_lut3(inputs, sks, params)
  temp_nodes[129] = outputs[0]
  temp_nodes[148] = outputs[1]
  temp_nodes[154] = outputs[2]
  temp_nodes[173] = outputs[3]
  inputs = [
      (
          temp_nodes[1912],
          temp_nodes[177],
          jaxite_bool.constant(False, params),
          8,
      ),
      (temp_nodes[179], temp_nodes[1922], my_string[119], 1),
      (temp_nodes[1764], temp_nodes[196], temp_nodes[191], 44),
      (
          temp_nodes[1800],
          temp_nodes[1763],
          jaxite_bool.constant(False, params),
          8,
      ),
  ]
  outputs = jaxite_bool.pmap_lut3(inputs, sks, params)
  temp_nodes[176] = outputs[0]
  temp_nodes[178] = outputs[1]
  temp_nodes[190] = outputs[2]
  temp_nodes[204] = outputs[3]
  inputs = [
      (temp_nodes[1903], my_string[22], jaxite_bool.constant(False, params), 1),
      (my_string[47], temp_nodes[1662], temp_nodes[304], 13),
      (temp_nodes[1912], temp_nodes[307], my_string[37], 128),
      (
          temp_nodes[1914],
          temp_nodes[1924],
          jaxite_bool.constant(False, params),
          8,
      ),
  ]
  outputs = jaxite_bool.pmap_lut3(inputs, sks, params)
  temp_nodes[297] = outputs[0]
  temp_nodes[303] = outputs[1]
  temp_nodes[306] = outputs[2]
  temp_nodes[310] = outputs[3]
  inputs = [
      (temp_nodes[1971], temp_nodes[359], temp_nodes[86], 64),
      (temp_nodes[317], temp_nodes[1975], temp_nodes[319], 128),
      (
          temp_nodes[83],
          temp_nodes[1660],
          jaxite_bool.constant(False, params),
          8,
      ),
      (
          temp_nodes[191],
          temp_nodes[196],
          jaxite_bool.constant(False, params),
          8,
      ),
  ]
  outputs = jaxite_bool.pmap_lut3(inputs, sks, params)
  temp_nodes[358] = outputs[0]
  temp_nodes[363] = outputs[1]
  temp_nodes[364] = outputs[2]
  temp_nodes[394] = outputs[3]
  inputs = [
      (my_string[183], temp_nodes[119], temp_nodes[1509], 16),
      (temp_nodes[153], my_string[187], my_string[190], 244),
      (temp_nodes[393], temp_nodes[1527], temp_nodes[1764], 176),
  ]
  outputs = jaxite_bool.pmap_lut3(inputs, sks, params)
  temp_nodes[1510] = outputs[0]
  temp_nodes[1515] = outputs[1]
  temp_nodes[1528] = outputs[2]
  inputs = [
      (temp_nodes[1615], my_string[92], jaxite_bool.constant(False, params), 8),
      (temp_nodes[1835], temp_nodes[1829], temp_nodes[1845], 112),
      (temp_nodes[1683], temp_nodes[1853], temp_nodes[1684], 112),
      (temp_nodes[1859], temp_nodes[1869], temp_nodes[1936], 64),
  ]
  outputs = jaxite_bool.pmap_lut3(inputs, sks, params)
  temp_nodes[1614] = outputs[0]
  temp_nodes[1828] = outputs[1]
  temp_nodes[1852] = outputs[2]
  temp_nodes[1858] = outputs[3]
  inputs = [
      (temp_nodes[1951], temp_nodes[2], temp_nodes[1957], 16),
      (
          temp_nodes[1970],
          temp_nodes[1972],
          jaxite_bool.constant(False, params),
          8,
      ),
      (temp_nodes[1951], temp_nodes[1891], temp_nodes[16], 16),
      (temp_nodes[1951], temp_nodes[31], temp_nodes[32], 64),
  ]
  outputs = jaxite_bool.pmap_lut3(inputs, sks, params)
  temp_nodes[1956] = outputs[0]
  temp_nodes[1969] = outputs[1]
  temp_nodes[15] = outputs[2]
  temp_nodes[30] = outputs[3]
  inputs = [
      (
          temp_nodes[1829],
          temp_nodes[1835],
          jaxite_bool.constant(False, params),
          8,
      ),
      (temp_nodes[54], temp_nodes[58], jaxite_bool.constant(False, params), 4),
      (temp_nodes[1829], temp_nodes[67], temp_nodes[1839], 96),
      (temp_nodes[71], temp_nodes[5], jaxite_bool.constant(False, params), 8),
  ]
  outputs = jaxite_bool.pmap_lut3(inputs, sks, params)
  temp_nodes[52] = outputs[0]
  temp_nodes[53] = outputs[1]
  temp_nodes[66] = outputs[2]
  temp_nodes[70] = outputs[3]
  inputs = [
      (temp_nodes[54], temp_nodes[111], jaxite_bool.constant(False, params), 6),
      (temp_nodes[113], temp_nodes[60], temp_nodes[61], 16),
      (temp_nodes[1510], temp_nodes[119], my_string[183], 1),
      (
          temp_nodes[122],
          temp_nodes[111],
          jaxite_bool.constant(False, params),
          8,
      ),
  ]
  outputs = jaxite_bool.pmap_lut3(inputs, sks, params)
  temp_nodes[110] = outputs[0]
  temp_nodes[112] = outputs[1]
  temp_nodes[115] = outputs[2]
  temp_nodes[121] = outputs[3]
  inputs = [
      (temp_nodes[1913], temp_nodes[1976], temp_nodes[129], 16),
      (my_string[39], my_string[47], jaxite_bool.constant(False, params), 14),
      (
          temp_nodes[1838],
          temp_nodes[148],
          jaxite_bool.constant(False, params),
          4,
      ),
      (
          temp_nodes[122],
          temp_nodes[111],
          jaxite_bool.constant(False, params),
          6,
      ),
  ]
  outputs = jaxite_bool.pmap_lut3(inputs, sks, params)
  temp_nodes[128] = outputs[0]
  temp_nodes[131] = outputs[1]
  temp_nodes[147] = outputs[2]
  temp_nodes[157] = outputs[3]
  inputs = [
      (
          temp_nodes[148],
          temp_nodes[1838],
          jaxite_bool.constant(False, params),
          8,
      ),
      (temp_nodes[173], temp_nodes[176], temp_nodes[178], 128),
      (
          temp_nodes[190],
          temp_nodes[1791],
          jaxite_bool.constant(False, params),
          8,
      ),
      (temp_nodes[193], temp_nodes[204], temp_nodes[1724], 96),
  ]
  outputs = jaxite_bool.pmap_lut3(inputs, sks, params)
  temp_nodes[158] = outputs[0]
  temp_nodes[172] = outputs[1]
  temp_nodes[189] = outputs[2]
  temp_nodes[203] = outputs[3]
  inputs = [
      (
          temp_nodes[1620],
          temp_nodes[1799],
          jaxite_bool.constant(False, params),
          8,
      ),
      (
          temp_nodes[1832],
          temp_nodes[1797],
          jaxite_bool.constant(False, params),
          8,
      ),
      (temp_nodes[1482], temp_nodes[1925], temp_nodes[1919], 64),
      (
          temp_nodes[1930],
          temp_nodes[303],
          jaxite_bool.constant(False, params),
          4,
      ),
  ]
  outputs = jaxite_bool.pmap_lut3(inputs, sks, params)
  temp_nodes[205] = outputs[0]
  temp_nodes[213] = outputs[1]
  temp_nodes[300] = outputs[2]
  temp_nodes[302] = outputs[3]
  inputs = [
      (temp_nodes[306], temp_nodes[310], my_string[34], 128),
      (temp_nodes[297], temp_nodes[1904], temp_nodes[331], 16),
      (temp_nodes[79], temp_nodes[358], temp_nodes[1931], 128),
      (temp_nodes[1486], temp_nodes[363], temp_nodes[364], 64),
  ]
  outputs = jaxite_bool.pmap_lut3(inputs, sks, params)
  temp_nodes[305] = outputs[0]
  temp_nodes[330] = outputs[1]
  temp_nodes[357] = outputs[2]
  temp_nodes[362] = outputs[3]
  inputs = [
      (temp_nodes[1942], temp_nodes[1801], temp_nodes[1734], 64),
      (my_string[111], my_string[30], my_string[31], 1),
      (my_string[191], temp_nodes[154], temp_nodes[1515], 16),
      (my_string[25], temp_nodes[1967], my_string[24], 16),
  ]
  outputs = jaxite_bool.pmap_lut3(inputs, sks, params)
  temp_nodes[549] = outputs[0]
  temp_nodes[1511] = outputs[1]
  temp_nodes[1516] = outputs[2]
  temp_nodes[1523] = outputs[3]
  inputs = [
      (temp_nodes[1887], temp_nodes[1966], temp_nodes[1900], 16),
      (temp_nodes[1528], temp_nodes[1724], temp_nodes[394], 248),
  ]
  outputs = jaxite_bool.pmap_lut3(inputs, sks, params)
  temp_nodes[1524] = outputs[0]
  temp_nodes[1529] = outputs[1]
  inputs = [
      (temp_nodes[1614], temp_nodes[1828], my_string[93], 128),
      (my_string[94], my_string[95], jaxite_bool.constant(False, params), 1),
      (temp_nodes[1852], temp_nodes[1951], temp_nodes[1858], 16),
      (temp_nodes[1852], temp_nodes[1859], temp_nodes[1956], 16),
  ]
  outputs = jaxite_bool.pmap_lut3(inputs, sks, params)
  temp_nodes[1613] = outputs[0]
  temp_nodes[1689] = outputs[1]
  temp_nodes[1851] = outputs[2]
  temp_nodes[1955] = outputs[3]
  inputs = [
      (temp_nodes[1852], temp_nodes[1859], temp_nodes[15], 16),
      (temp_nodes[1852], temp_nodes[1859], temp_nodes[30], 16),
      (temp_nodes[66], temp_nodes[53], temp_nodes[52], 7),
      (temp_nodes[1848], temp_nodes[1823], temp_nodes[70], 16),
  ]
  outputs = jaxite_bool.pmap_lut3(inputs, sks, params)
  temp_nodes[14] = outputs[0]
  temp_nodes[29] = outputs[1]
  temp_nodes[51] = outputs[2]
  temp_nodes[69] = outputs[3]
  inputs = [
      (
          temp_nodes[1873],
          my_string[100],
          jaxite_bool.constant(False, params),
          4,
      ),
      (
          temp_nodes[357],
          temp_nodes[362],
          jaxite_bool.constant(False, params),
          8,
      ),
      (temp_nodes[110], temp_nodes[115], temp_nodes[112], 16),
      (temp_nodes[66], temp_nodes[121], temp_nodes[61], 96),
  ]
  outputs = jaxite_bool.pmap_lut3(inputs, sks, params)
  temp_nodes[72] = outputs[0]
  temp_nodes[78] = outputs[1]
  temp_nodes[109] = outputs[2]
  temp_nodes[120] = outputs[3]
  inputs = [
      (temp_nodes[82], temp_nodes[80], temp_nodes[83], 64),
      (
          temp_nodes[147],
          temp_nodes[112],
          jaxite_bool.constant(False, params),
          1,
      ),
      (temp_nodes[61], temp_nodes[60], temp_nodes[115], 208),
      (temp_nodes[1516], temp_nodes[154], my_string[191], 1),
  ]
  outputs = jaxite_bool.pmap_lut3(inputs, sks, params)
  temp_nodes[132] = outputs[0]
  temp_nodes[146] = outputs[1]
  temp_nodes[149] = outputs[2]
  temp_nodes[150] = outputs[3]
  inputs = [
      (temp_nodes[61], temp_nodes[157], temp_nodes[158], 7),
      (
          temp_nodes[157],
          temp_nodes[158],
          jaxite_bool.constant(False, params),
          8,
      ),
      (
          temp_nodes[1496],
          temp_nodes[1494],
          jaxite_bool.constant(False, params),
          1,
      ),
      (temp_nodes[1484], temp_nodes[1482], temp_nodes[172], 16),
  ]
  outputs = jaxite_bool.pmap_lut3(inputs, sks, params)
  temp_nodes[156] = outputs[0]
  temp_nodes[159] = outputs[1]
  temp_nodes[170] = outputs[2]
  temp_nodes[171] = outputs[3]
  inputs = [
      (
          temp_nodes[203],
          temp_nodes[205],
          jaxite_bool.constant(False, params),
          6,
      ),
      (temp_nodes[189], temp_nodes[213], temp_nodes[1809], 96),
      (temp_nodes[61], temp_nodes[110], temp_nodes[158], 7),
      (my_string[194], my_string[193], jaxite_bool.constant(False, params), 1),
  ]
  outputs = jaxite_bool.pmap_lut3(inputs, sks, params)
  temp_nodes[202] = outputs[0]
  temp_nodes[212] = outputs[1]
  temp_nodes[258] = outputs[2]
  temp_nodes[263] = outputs[3]
  inputs = [
      (temp_nodes[1922], temp_nodes[1931], temp_nodes[300], 64),
      (temp_nodes[1490], temp_nodes[302], temp_nodes[305], 64),
      (
          temp_nodes[549],
          temp_nodes[1708],
          jaxite_bool.constant(False, params),
          8,
      ),
      (temp_nodes[1935], temp_nodes[131], temp_nodes[1511], 16),
  ]
  outputs = jaxite_bool.pmap_lut3(inputs, sks, params)
  temp_nodes[299] = outputs[0]
  temp_nodes[301] = outputs[1]
  temp_nodes[548] = outputs[2]
  temp_nodes[1512] = outputs[3]
  inputs = [
      (temp_nodes[1494], temp_nodes[128], temp_nodes[1969], 64),
      (temp_nodes[1523], temp_nodes[1524], temp_nodes[330], 128),
      (
          temp_nodes[1529],
          temp_nodes[1791],
          jaxite_bool.constant(False, params),
          8,
      ),
      (temp_nodes[193], temp_nodes[204], temp_nodes[205], 23),
  ]
  outputs = jaxite_bool.pmap_lut3(inputs, sks, params)
  temp_nodes[1513] = outputs[0]
  temp_nodes[1525] = outputs[1]
  temp_nodes[1530] = outputs[2]
  temp_nodes[1534] = outputs[3]
  inputs = [
      (temp_nodes[1689], temp_nodes[1613], temp_nodes[1851], 208),
      (my_string[102], my_string[103], jaxite_bool.constant(False, params), 1),
      (temp_nodes[1689], temp_nodes[1613], temp_nodes[14], 208),
      (
          temp_nodes[29],
          temp_nodes[1945],
          jaxite_bool.constant(False, params),
          4,
      ),
  ]
  outputs = jaxite_bool.pmap_lut3(inputs, sks, params)
  temp_nodes[1612] = outputs[0]
  temp_nodes[1694] = outputs[1]
  temp_nodes[13] = outputs[2]
  temp_nodes[28] = outputs[3]
  inputs = [
      (my_string[41], temp_nodes[1884], jaxite_bool.constant(False, params), 4),
      (temp_nodes[1884], my_string[40], jaxite_bool.constant(False, params), 8),
      (temp_nodes[51], temp_nodes[69], temp_nodes[72], 128),
      (
          temp_nodes[1613],
          temp_nodes[1689],
          jaxite_bool.constant(False, params),
          4,
      ),
  ]
  outputs = jaxite_bool.pmap_lut3(inputs, sks, params)
  temp_nodes[45] = outputs[0]
  temp_nodes[47] = outputs[1]
  temp_nodes[50] = outputs[2]
  temp_nodes[75] = outputs[3]
  inputs = [
      (temp_nodes[1955], temp_nodes[2], temp_nodes[1963], 1),
      (temp_nodes[1870], temp_nodes[1893], temp_nodes[78], 16),
      (
          temp_nodes[109],
          temp_nodes[120],
          jaxite_bool.constant(False, params),
          8,
      ),
      (temp_nodes[150], temp_nodes[149], temp_nodes[146], 64),
  ]
  outputs = jaxite_bool.pmap_lut3(inputs, sks, params)
  temp_nodes[76] = outputs[0]
  temp_nodes[77] = outputs[1]
  temp_nodes[108] = outputs[2]
  temp_nodes[145] = outputs[3]
  inputs = [
      (temp_nodes[156], temp_nodes[159], temp_nodes[115], 16),
      (temp_nodes[170], temp_nodes[171], temp_nodes[1970], 128),
      (temp_nodes[1492], temp_nodes[1490], temp_nodes[1486], 1),
      (
          temp_nodes[1938],
          temp_nodes[1889],
          jaxite_bool.constant(False, params),
          1,
      ),
  ]
  outputs = jaxite_bool.pmap_lut3(inputs, sks, params)
  temp_nodes[155] = outputs[0]
  temp_nodes[169] = outputs[1]
  temp_nodes[180] = outputs[2]
  temp_nodes[182] = outputs[3]
  inputs = [
      (temp_nodes[1882], my_string[81], temp_nodes[1885], 16),
      (
          temp_nodes[202],
          temp_nodes[1764],
          jaxite_bool.constant(False, params),
          8,
      ),
      (
          temp_nodes[202],
          temp_nodes[196],
          jaxite_bool.constant(False, params),
          8,
      ),
      (temp_nodes[212], temp_nodes[55], temp_nodes[1839], 96),
  ]
  outputs = jaxite_bool.pmap_lut3(inputs, sks, params)
  temp_nodes[183] = outputs[0]
  temp_nodes[201] = outputs[1]
  temp_nodes[217] = outputs[2]
  temp_nodes[243] = outputs[3]
  inputs = [
      (temp_nodes[54], temp_nodes[111], jaxite_bool.constant(False, params), 8),
      (
          temp_nodes[110],
          temp_nodes[158],
          jaxite_bool.constant(False, params),
          8,
      ),
      (
          temp_nodes[258],
          temp_nodes[115],
          jaxite_bool.constant(False, params),
          4,
      ),
      (my_string[197], my_string[196], my_string[198], 7),
  ]
  outputs = jaxite_bool.pmap_lut3(inputs, sks, params)
  temp_nodes[246] = outputs[0]
  temp_nodes[249] = outputs[1]
  temp_nodes[257] = outputs[2]
  temp_nodes[264] = outputs[3]
  inputs = [
      (
          temp_nodes[147],
          temp_nodes[115],
          jaxite_bool.constant(False, params),
          8,
      ),
      (
          temp_nodes[1915],
          temp_nodes[84],
          jaxite_bool.constant(False, params),
          1,
      ),
      (my_string[127], my_string[119], jaxite_bool.constant(False, params), 1),
      (temp_nodes[22], temp_nodes[299], temp_nodes[301], 128),
  ]
  outputs = jaxite_bool.pmap_lut3(inputs, sks, params)
  temp_nodes[267] = outputs[0]
  temp_nodes[277] = outputs[1]
  temp_nodes[280] = outputs[2]
  temp_nodes[298] = outputs[3]
  inputs = [
      (my_string[47], temp_nodes[1662], temp_nodes[130], 13),
      (temp_nodes[212], temp_nodes[55], jaxite_bool.constant(False, params), 8),
      (temp_nodes[213], temp_nodes[189], temp_nodes[1530], 7),
      (temp_nodes[1512], temp_nodes[1513], temp_nodes[132], 128),
  ]
  outputs = jaxite_bool.pmap_lut3(inputs, sks, params)
  temp_nodes[333] = outputs[0]
  temp_nodes[391] = outputs[1]
  temp_nodes[392] = outputs[2]
  temp_nodes[1514] = outputs[3]
  inputs = [
      (temp_nodes[263], my_string[195], my_string[198], 244),
      (my_string[33], my_string[36], my_string[32], 16),
      (temp_nodes[2], temp_nodes[1958], temp_nodes[1525], 64),
      (temp_nodes[548], temp_nodes[1534], temp_nodes[1764], 176),
  ]
  outputs = jaxite_bool.pmap_lut3(inputs, sks, params)
  temp_nodes[1517] = outputs[0]
  temp_nodes[1519] = outputs[1]
  temp_nodes[1526] = outputs[2]
  temp_nodes[1535] = outputs[3]
  inputs = [
      (
          temp_nodes[1612],
          temp_nodes[1955],
          jaxite_bool.constant(False, params),
          1,
      ),
      (temp_nodes[5], temp_nodes[1862], jaxite_bool.constant(False, params), 8),
      (my_string[65], my_string[66], my_string[69], 64),
      (temp_nodes[1612], temp_nodes[1955], temp_nodes[1852], 1),
  ]
  outputs = jaxite_bool.pmap_lut3(inputs, sks, params)
  temp_nodes[1611] = outputs[0]
  temp_nodes[4] = outputs[1]
  temp_nodes[8] = outputs[2]
  temp_nodes[11] = outputs[3]
  inputs = [
      (temp_nodes[13], temp_nodes[28], temp_nodes[45], 64),
      (temp_nodes[13], temp_nodes[28], temp_nodes[47], 64),
      (my_string[101], temp_nodes[50], temp_nodes[1694], 112),
      (
          temp_nodes[1612],
          temp_nodes[1897],
          jaxite_bool.constant(False, params),
          1,
      ),
  ]
  outputs = jaxite_bool.pmap_lut3(inputs, sks, params)
  temp_nodes[12] = outputs[0]
  temp_nodes[46] = outputs[1]
  temp_nodes[49] = outputs[2]
  temp_nodes[73] = outputs[3]
  inputs = [
      (temp_nodes[75], temp_nodes[76], temp_nodes[77], 64),
      (temp_nodes[13], temp_nodes[29], temp_nodes[1903], 16),
      (temp_nodes[13], temp_nodes[29], temp_nodes[1951], 254),
      (temp_nodes[13], temp_nodes[29], jaxite_bool.constant(False, params), 1),
  ]
  outputs = jaxite_bool.pmap_lut3(inputs, sks, params)
  temp_nodes[74] = outputs[0]
  temp_nodes[88] = outputs[1]
  temp_nodes[89] = outputs[2]
  temp_nodes[94] = outputs[3]
  inputs = [
      (temp_nodes[108], temp_nodes[51], jaxite_bool.constant(False, params), 4),
      (
          temp_nodes[70],
          temp_nodes[1863],
          jaxite_bool.constant(False, params),
          8,
      ),
      (temp_nodes[1958], temp_nodes[1514], temp_nodes[79], 128),
      (temp_nodes[13], temp_nodes[29], temp_nodes[1860], 16),
  ]
  outputs = jaxite_bool.pmap_lut3(inputs, sks, params)
  temp_nodes[107] = outputs[0]
  temp_nodes[124] = outputs[1]
  temp_nodes[127] = outputs[2]
  temp_nodes[135] = outputs[3]
  inputs = [
      (
          temp_nodes[155],
          temp_nodes[145],
          jaxite_bool.constant(False, params),
          4,
      ),
      (temp_nodes[120], temp_nodes[159], temp_nodes[115], 96),
      (temp_nodes[1868], temp_nodes[1818], temp_nodes[71], 16),
      (temp_nodes[1936], temp_nodes[169], temp_nodes[180], 128),
  ]
  outputs = jaxite_bool.pmap_lut3(inputs, sks, params)
  temp_nodes[144] = outputs[0]
  temp_nodes[160] = outputs[1]
  temp_nodes[162] = outputs[2]
  temp_nodes[168] = outputs[3]
  inputs = [
      (
          temp_nodes[182],
          temp_nodes[183],
          jaxite_bool.constant(False, params),
          8,
      ),
      (
          temp_nodes[212],
          temp_nodes[1835],
          jaxite_bool.constant(False, params),
          8,
      ),
      (temp_nodes[243], temp_nodes[53], jaxite_bool.constant(False, params), 8),
      (temp_nodes[243], temp_nodes[246], temp_nodes[61], 96),
  ]
  outputs = jaxite_bool.pmap_lut3(inputs, sks, params)
  temp_nodes[181] = outputs[0]
  temp_nodes[211] = outputs[1]
  temp_nodes[242] = outputs[2]
  temp_nodes[245] = outputs[3]
  inputs = [
      (
          temp_nodes[249],
          temp_nodes[257],
          jaxite_bool.constant(False, params),
          4,
      ),
      (
          temp_nodes[75],
          temp_nodes[1936],
          jaxite_bool.constant(False, params),
          4,
      ),
      (
          temp_nodes[1859],
          temp_nodes[297],
          jaxite_bool.constant(False, params),
          1,
      ),
      (temp_nodes[1891], temp_nodes[1963], temp_nodes[1526], 16),
  ]
  outputs = jaxite_bool.pmap_lut3(inputs, sks, params)
  temp_nodes[256] = outputs[0]
  temp_nodes[294] = outputs[1]
  temp_nodes[296] = outputs[2]
  temp_nodes[329] = outputs[3]
  inputs = [
      (temp_nodes[132], temp_nodes[128], temp_nodes[333], 128),
      (
          temp_nodes[1934],
          temp_nodes[323],
          jaxite_bool.constant(False, params),
          1,
      ),
      (temp_nodes[309], my_string[111], my_string[31], 1),
      (
          temp_nodes[243],
          temp_nodes[246],
          jaxite_bool.constant(False, params),
          8,
      ),
  ]
  outputs = jaxite_bool.pmap_lut3(inputs, sks, params)
  temp_nodes[374] = outputs[0]
  temp_nodes[376] = outputs[1]
  temp_nodes[377] = outputs[2]
  temp_nodes[389] = outputs[3]
  inputs = [
      (temp_nodes[1809], temp_nodes[392], temp_nodes[391], 13),
      (
          temp_nodes[156],
          temp_nodes[267],
          jaxite_bool.constant(False, params),
          4,
      ),
      (temp_nodes[175], temp_nodes[277], temp_nodes[280], 64),
      (my_string[199], temp_nodes[264], temp_nodes[1517], 16),
  ]
  outputs = jaxite_bool.pmap_lut3(inputs, sks, params)
  temp_nodes[390] = outputs[0]
  temp_nodes[399] = outputs[1]
  temp_nodes[607] = outputs[2]
  temp_nodes[1518] = outputs[3]
  inputs = [
      (temp_nodes[1897], temp_nodes[1519], temp_nodes[298], 64),
      (temp_nodes[1535], temp_nodes[1724], temp_nodes[217], 7),
      (temp_nodes[201], temp_nodes[196], temp_nodes[213], 31),
  ]
  outputs = jaxite_bool.pmap_lut3(inputs, sks, params)
  temp_nodes[1520] = outputs[0]
  temp_nodes[1577] = outputs[1]
  temp_nodes[1578] = outputs[2]
  inputs = [
      (temp_nodes[1611], temp_nodes[4], my_string[64], 128),
      (my_string[110], my_string[111], jaxite_bool.constant(False, params), 1),
      (temp_nodes[1611], temp_nodes[4], my_string[68], 128),
      (temp_nodes[1674], temp_nodes[8], my_string[67], 128),
  ]
  outputs = jaxite_bool.pmap_lut3(inputs, sks, params)
  temp_nodes[1610] = outputs[0]
  temp_nodes[1699] = outputs[1]
  temp_nodes[6] = outputs[2]
  temp_nodes[7] = outputs[3]
  inputs = [
      (temp_nodes[11], temp_nodes[12], temp_nodes[46], 128),
      (temp_nodes[49], temp_nodes[73], temp_nodes[74], 64),
      (my_string[22], temp_nodes[88], temp_nodes[89], 14),
      (temp_nodes[107], temp_nodes[124], temp_nodes[1700], 128),
  ]
  outputs = jaxite_bool.pmap_lut3(inputs, sks, params)
  temp_nodes[10] = outputs[0]
  temp_nodes[48] = outputs[1]
  temp_nodes[87] = outputs[2]
  temp_nodes[106] = outputs[3]
  inputs = [
      (temp_nodes[11], temp_nodes[73], temp_nodes[127], 128),
      (temp_nodes[1892], temp_nodes[94], temp_nodes[1660], 112),
      (
          temp_nodes[135],
          temp_nodes[1674],
          jaxite_bool.constant(False, params),
          4,
      ),
      (
          temp_nodes[144],
          temp_nodes[160],
          jaxite_bool.constant(False, params),
          8,
      ),
  ]
  outputs = jaxite_bool.pmap_lut3(inputs, sks, params)
  temp_nodes[126] = outputs[0]
  temp_nodes[133] = outputs[1]
  temp_nodes[134] = outputs[2]
  temp_nodes[143] = outputs[3]
  inputs = [
      (temp_nodes[162], temp_nodes[168], temp_nodes[181], 128),
      (
          temp_nodes[189],
          temp_nodes[1796],
          jaxite_bool.constant(False, params),
          8,
      ),
      (
          temp_nodes[192],
          temp_nodes[1815],
          jaxite_bool.constant(False, params),
          8,
      ),
      (
          temp_nodes[1873],
          temp_nodes[1885],
          jaxite_bool.constant(False, params),
          4,
      ),
  ]
  outputs = jaxite_bool.pmap_lut3(inputs, sks, params)
  temp_nodes[167] = outputs[0]
  temp_nodes[188] = outputs[1]
  temp_nodes[197] = outputs[2]
  temp_nodes[198] = outputs[3]
  inputs = [
      (temp_nodes[1798], temp_nodes[201], temp_nodes[1847], 64),
      (
          temp_nodes[242],
          temp_nodes[211],
          jaxite_bool.constant(False, params),
          1,
      ),
      (
          temp_nodes[245],
          temp_nodes[109],
          jaxite_bool.constant(False, params),
          8,
      ),
      (temp_nodes[245], temp_nodes[249], temp_nodes[115], 96),
  ]
  outputs = jaxite_bool.pmap_lut3(inputs, sks, params)
  temp_nodes[200] = outputs[0]
  temp_nodes[241] = outputs[1]
  temp_nodes[244] = outputs[2]
  temp_nodes[248] = outputs[3]
  inputs = [
      (
          temp_nodes[256],
          temp_nodes[147],
          jaxite_bool.constant(False, params),
          8,
      ),
      (
          temp_nodes[149],
          temp_nodes[150],
          jaxite_bool.constant(False, params),
          4,
      ),
      (temp_nodes[1518], temp_nodes[264], my_string[199], 1),
      (
          temp_nodes[256],
          temp_nodes[267],
          jaxite_bool.constant(False, params),
          1,
      ),
  ]
  outputs = jaxite_bool.pmap_lut3(inputs, sks, params)
  temp_nodes[255] = outputs[0]
  temp_nodes[259] = outputs[1]
  temp_nodes[260] = outputs[2]
  temp_nodes[266] = outputs[3]
  inputs = [
      (
          temp_nodes[148],
          temp_nodes[115],
          jaxite_bool.constant(False, params),
          4,
      ),
      (
          temp_nodes[1928],
          temp_nodes[81],
          jaxite_bool.constant(False, params),
          4,
      ),
      (my_string[79], temp_nodes[179], jaxite_bool.constant(False, params), 1),
      (temp_nodes[1951], temp_nodes[1904], temp_nodes[294], 16),
  ]
  outputs = jaxite_bool.pmap_lut3(inputs, sks, params)
  temp_nodes[268] = outputs[0]
  temp_nodes[278] = outputs[1]
  temp_nodes[279] = outputs[2]
  temp_nodes[293] = outputs[3]
  inputs = [
      (temp_nodes[1852], temp_nodes[1891], temp_nodes[296], 16),
      (my_string[31], my_string[23], my_string[43], 16),
      (temp_nodes[1852], temp_nodes[1951], temp_nodes[329], 16),
      (temp_nodes[79], temp_nodes[374], temp_nodes[1970], 128),
  ]
  outputs = jaxite_bool.pmap_lut3(inputs, sks, params)
  temp_nodes[295] = outputs[0]
  temp_nodes[320] = outputs[1]
  temp_nodes[328] = outputs[2]
  temp_nodes[373] = outputs[3]
  inputs = [
      (temp_nodes[1494], temp_nodes[376], temp_nodes[377], 64),
      (
          temp_nodes[245],
          temp_nodes[249],
          jaxite_bool.constant(False, params),
          8,
      ),
      (temp_nodes[1839], temp_nodes[390], temp_nodes[389], 13),
      (
          temp_nodes[399],
          temp_nodes[150],
          jaxite_bool.constant(False, params),
          4,
      ),
  ]
  outputs = jaxite_bool.pmap_lut3(inputs, sks, params)
  temp_nodes[375] = outputs[0]
  temp_nodes[387] = outputs[1]
  temp_nodes[388] = outputs[2]
  temp_nodes[398] = outputs[3]
  inputs = [
      (temp_nodes[115], temp_nodes[146], temp_nodes[150], 208),
      (temp_nodes[1494], temp_nodes[1930], temp_nodes[607], 16),
      (temp_nodes[1870], temp_nodes[1520], temp_nodes[1894], 64),
      (temp_nodes[1578], temp_nodes[1577], temp_nodes[1809], 112),
  ]
  outputs = jaxite_bool.pmap_lut3(inputs, sks, params)
  temp_nodes[405] = outputs[0]
  temp_nodes[606] = outputs[1]
  temp_nodes[1521] = outputs[2]
  temp_nodes[1579] = outputs[3]
  inputs = [
      (temp_nodes[6], temp_nodes[1610], temp_nodes[7], 64),
      (temp_nodes[1679], my_string[75], jaxite_bool.constant(False, params), 8),
      (temp_nodes[10], temp_nodes[48], temp_nodes[87], 128),
      (
          temp_nodes[94],
          temp_nodes[1952],
          jaxite_bool.constant(False, params),
          8,
      ),
  ]
  outputs = jaxite_bool.pmap_lut3(inputs, sks, params)
  temp_nodes[1609] = outputs[0]
  temp_nodes[1678] = outputs[1]
  temp_nodes[9] = outputs[2]
  temp_nodes[93] = outputs[3]
  inputs = [
      (
          temp_nodes[1955],
          temp_nodes[1963],
          jaxite_bool.constant(False, params),
          1,
      ),
      (temp_nodes[50], my_string[101], jaxite_bool.constant(False, params), 8),
      (
          temp_nodes[106],
          temp_nodes[1699],
          jaxite_bool.constant(False, params),
          4,
      ),
      (temp_nodes[133], temp_nodes[134], temp_nodes[126], 16),
  ]
  outputs = jaxite_bool.pmap_lut3(inputs, sks, params)
  temp_nodes[100] = outputs[0]
  temp_nodes[103] = outputs[1]
  temp_nodes[105] = outputs[2]
  temp_nodes[125] = outputs[3]
  inputs = [
      (
          temp_nodes[143],
          temp_nodes[107],
          jaxite_bool.constant(False, params),
          4,
      ),
      (temp_nodes[162], my_string[116], my_string[117], 128),
      (temp_nodes[49], temp_nodes[89], temp_nodes[167], 16),
      (temp_nodes[188], temp_nodes[197], temp_nodes[198], 16),
  ]
  outputs = jaxite_bool.pmap_lut3(inputs, sks, params)
  temp_nodes[142] = outputs[0]
  temp_nodes[161] = outputs[1]
  temp_nodes[166] = outputs[2]
  temp_nodes[187] = outputs[3]
  inputs = [
      (
          temp_nodes[200],
          temp_nodes[1882],
          jaxite_bool.constant(False, params),
          1,
      ),
      (temp_nodes[196], temp_nodes[201], temp_nodes[1791], 224),
      (
          temp_nodes[197],
          temp_nodes[1938],
          jaxite_bool.constant(False, params),
          1,
      ),
      (
          temp_nodes[244],
          temp_nodes[241],
          jaxite_bool.constant(False, params),
          4,
      ),
  ]
  outputs = jaxite_bool.pmap_lut3(inputs, sks, params)
  temp_nodes[199] = outputs[0]
  temp_nodes[216] = outputs[1]
  temp_nodes[219] = outputs[2]
  temp_nodes[240] = outputs[3]
  inputs = [
      (
          temp_nodes[248],
          temp_nodes[144],
          jaxite_bool.constant(False, params),
          8,
      ),
      (temp_nodes[191], temp_nodes[1846], temp_nodes[1764], 128),
      (
          temp_nodes[248],
          temp_nodes[255],
          jaxite_bool.constant(False, params),
          6,
      ),
      (temp_nodes[255], temp_nodes[266], temp_nodes[150], 16),
  ]
  outputs = jaxite_bool.pmap_lut3(inputs, sks, params)
  temp_nodes[247] = outputs[0]
  temp_nodes[251] = outputs[1]
  temp_nodes[254] = outputs[2]
  temp_nodes[265] = outputs[3]
  inputs = [
      (
          temp_nodes[1496],
          temp_nodes[1498],
          jaxite_bool.constant(False, params),
          1,
      ),
      (temp_nodes[130], temp_nodes[319], temp_nodes[320], 64),
      (temp_nodes[308], temp_nodes[323], temp_nodes[1975], 16),
      (temp_nodes[1660], my_string[45], jaxite_bool.constant(False, params), 8),
  ]
  outputs = jaxite_bool.pmap_lut3(inputs, sks, params)
  temp_nodes[275] = outputs[0]
  temp_nodes[318] = outputs[1]
  temp_nodes[322] = outputs[2]
  temp_nodes[324] = outputs[3]
  inputs = [
      (
          temp_nodes[1859],
          temp_nodes[328],
          jaxite_bool.constant(False, params),
          4,
      ),
      (temp_nodes[1958], temp_nodes[373], temp_nodes[375], 128),
      (temp_nodes[61], temp_nodes[388], temp_nodes[387], 13),
      (
          temp_nodes[266],
          temp_nodes[398],
          jaxite_bool.constant(False, params),
          4,
      ),
  ]
  outputs = jaxite_bool.pmap_lut3(inputs, sks, params)
  temp_nodes[327] = outputs[0]
  temp_nodes[372] = outputs[1]
  temp_nodes[386] = outputs[2]
  temp_nodes[397] = outputs[3]
  inputs = [
      (
          temp_nodes[268],
          temp_nodes[259],
          jaxite_bool.constant(False, params),
          8,
      ),
      (my_string[202], my_string[201], my_string[206], 1),
      (my_string[203], my_string[204], my_string[205], 128),
      (
          temp_nodes[405],
          temp_nodes[260],
          jaxite_bool.constant(False, params),
          8,
      ),
  ]
  outputs = jaxite_bool.pmap_lut3(inputs, sks, params)
  temp_nodes[400] = outputs[0]
  temp_nodes[408] = outputs[1]
  temp_nodes[410] = outputs[2]
  temp_nodes[453] = outputs[3]
  inputs = [
      (my_string[212], my_string[213], jaxite_bool.constant(False, params), 8),
      (temp_nodes[606], temp_nodes[86], temp_nodes[1921], 128),
      (my_string[63], temp_nodes[279], temp_nodes[278], 64),
      (temp_nodes[1521], temp_nodes[293], temp_nodes[295], 128),
  ]
  outputs = jaxite_bool.pmap_lut3(inputs, sks, params)
  temp_nodes[461] = outputs[0]
  temp_nodes[605] = outputs[1]
  temp_nodes[608] = outputs[2]
  temp_nodes[1522] = outputs[3]
  inputs = [
      (my_string[208], my_string[215], my_string[209], 7),
      (temp_nodes[1791], temp_nodes[1579], temp_nodes[391], 7),
  ]
  outputs = jaxite_bool.pmap_lut3(inputs, sks, params)
  temp_nodes[1531] = outputs[0]
  temp_nodes[1580] = outputs[1]
  inputs = [
      (temp_nodes[9], temp_nodes[1609], temp_nodes[1612], 7),
      (my_string[118], my_string[119], jaxite_bool.constant(False, params), 1),
      (
          temp_nodes[1845],
          temp_nodes[1953],
          jaxite_bool.constant(False, params),
          8,
      ),
      (temp_nodes[93], temp_nodes[1678], my_string[77], 64),
  ]
  outputs = jaxite_bool.pmap_lut3(inputs, sks, params)
  temp_nodes[1608] = outputs[0]
  temp_nodes[1705] = outputs[1]
  temp_nodes[90] = outputs[2]
  temp_nodes[92] = outputs[3]
  inputs = [
      (temp_nodes[1609], temp_nodes[9], jaxite_bool.constant(False, params), 8),
      (temp_nodes[1955], temp_nodes[2], jaxite_bool.constant(False, params), 1),
      (temp_nodes[9], temp_nodes[1609], temp_nodes[103], 112),
      (
          temp_nodes[105],
          temp_nodes[125],
          jaxite_bool.constant(False, params),
          4,
      ),
  ]
  outputs = jaxite_bool.pmap_lut3(inputs, sks, params)
  temp_nodes[98] = outputs[0]
  temp_nodes[99] = outputs[1]
  temp_nodes[102] = outputs[2]
  temp_nodes[104] = outputs[3]
  inputs = [
      (temp_nodes[9], temp_nodes[1609], temp_nodes[100], 112),
      (temp_nodes[1826], temp_nodes[161], temp_nodes[142], 64),
      (temp_nodes[105], temp_nodes[134], temp_nodes[166], 16),
      (temp_nodes[187], temp_nodes[199], temp_nodes[182], 128),
  ]
  outputs = jaxite_bool.pmap_lut3(inputs, sks, params)
  temp_nodes[140] = outputs[0]
  temp_nodes[141] = outputs[1]
  temp_nodes[165] = outputs[2]
  temp_nodes[186] = outputs[3]
  inputs = [
      (my_string[82], my_string[80], my_string[85], 128),
      (
          temp_nodes[217],
          temp_nodes[216],
          jaxite_bool.constant(False, params),
          4,
      ),
      (
          temp_nodes[247],
          temp_nodes[240],
          jaxite_bool.constant(False, params),
          4,
      ),
      (temp_nodes[188], temp_nodes[251], temp_nodes[219], 16),
  ]
  outputs = jaxite_bool.pmap_lut3(inputs, sks, params)
  temp_nodes[207] = outputs[0]
  temp_nodes[215] = outputs[1]
  temp_nodes[239] = outputs[2]
  temp_nodes[250] = outputs[3]
  inputs = [
      (temp_nodes[260], temp_nodes[259], temp_nodes[254], 64),
      (
          temp_nodes[1492],
          temp_nodes[1908],
          jaxite_bool.constant(False, params),
          4,
      ),
      (
          temp_nodes[1522],
          temp_nodes[75],
          jaxite_bool.constant(False, params),
          1,
      ),
      (
          temp_nodes[317],
          temp_nodes[318],
          jaxite_bool.constant(False, params),
          8,
      ),
  ]
  outputs = jaxite_bool.pmap_lut3(inputs, sks, params)
  temp_nodes[253] = outputs[0]
  temp_nodes[274] = outputs[1]
  temp_nodes[292] = outputs[2]
  temp_nodes[316] = outputs[3]
  inputs = [
      (temp_nodes[322], temp_nodes[324], my_string[42], 128),
      (
          temp_nodes[1522],
          temp_nodes[327],
          jaxite_bool.constant(False, params),
          1,
      ),
      (temp_nodes[88], my_string[22], jaxite_bool.constant(False, params), 1),
      (temp_nodes[1870], temp_nodes[357], temp_nodes[362], 64),
  ]
  outputs = jaxite_bool.pmap_lut3(inputs, sks, params)
  temp_nodes[321] = outputs[0]
  temp_nodes[326] = outputs[1]
  temp_nodes[336] = outputs[2]
  temp_nodes[356] = outputs[3]
  inputs = [
      (temp_nodes[133], temp_nodes[134], temp_nodes[73], 16),
      (temp_nodes[11], temp_nodes[372], temp_nodes[1972], 128),
      (
          temp_nodes[386],
          temp_nodes[115],
          jaxite_bool.constant(False, params),
          4,
      ),
      (temp_nodes[248], temp_nodes[257], temp_nodes[147], 128),
  ]
  outputs = jaxite_bool.pmap_lut3(inputs, sks, params)
  temp_nodes[370] = outputs[0]
  temp_nodes[371] = outputs[1]
  temp_nodes[385] = outputs[2]
  temp_nodes[395] = outputs[3]
  inputs = [
      (
          temp_nodes[397],
          temp_nodes[400],
          jaxite_bool.constant(False, params),
          8,
      ),
      (temp_nodes[267], temp_nodes[155], temp_nodes[398], 224),
      (my_string[206], temp_nodes[410], temp_nodes[408], 241),
      (my_string[205], my_string[204], my_string[206], 7),
  ]
  outputs = jaxite_bool.pmap_lut3(inputs, sks, params)
  temp_nodes[396] = outputs[0]
  temp_nodes[402] = outputs[1]
  temp_nodes[407] = outputs[2]
  temp_nodes[411] = outputs[3]
  inputs = [
      (
          temp_nodes[160],
          temp_nodes[399],
          jaxite_bool.constant(False, params),
          6,
      ),
      (my_string[135], my_string[127], jaxite_bool.constant(False, params), 1),
      (temp_nodes[397], temp_nodes[400], temp_nodes[260], 96),
      (
          temp_nodes[453],
          temp_nodes[149],
          jaxite_bool.constant(False, params),
          8,
      ),
  ]
  outputs = jaxite_bool.pmap_lut3(inputs, sks, params)
  temp_nodes[416] = outputs[0]
  temp_nodes[430] = outputs[1]
  temp_nodes[451] = outputs[2]
  temp_nodes[452] = outputs[3]
  inputs = [
      (
          temp_nodes[265],
          temp_nodes[400],
          jaxite_bool.constant(False, params),
          6,
      ),
      (
          temp_nodes[265],
          temp_nodes[400],
          jaxite_bool.constant(False, params),
          8,
      ),
      (
          temp_nodes[1730],
          my_string[135],
          jaxite_bool.constant(False, params),
          4,
      ),
      (temp_nodes[1839], temp_nodes[1580], temp_nodes[389], 13),
  ]
  outputs = jaxite_bool.pmap_lut3(inputs, sks, params)
  temp_nodes[467] = outputs[0]
  temp_nodes[469] = outputs[1]
  temp_nodes[489] = outputs[2]
  temp_nodes[547] = outputs[3]
  inputs = [
      (temp_nodes[605], temp_nodes[275], temp_nodes[608], 128),
      (temp_nodes[1531], my_string[210], temp_nodes[461], 208),
  ]
  outputs = jaxite_bool.pmap_lut3(inputs, sks, params)
  temp_nodes[604] = outputs[0]
  temp_nodes[1532] = outputs[1]
  inputs = [
      (temp_nodes[1955], temp_nodes[1608], temp_nodes[90], 64),
      (temp_nodes[92], my_string[74], my_string[72], 128),
      (
          temp_nodes[75],
          temp_nodes[1608],
          jaxite_bool.constant(False, params),
          4,
      ),
      (temp_nodes[98], temp_nodes[99], temp_nodes[100], 64),
  ]
  outputs = jaxite_bool.pmap_lut3(inputs, sks, params)
  temp_nodes[1607] = outputs[0]
  temp_nodes[91] = outputs[1]
  temp_nodes[96] = outputs[2]
  temp_nodes[97] = outputs[3]
  inputs = [
      (temp_nodes[1694], temp_nodes[102], temp_nodes[104], 208),
      (temp_nodes[1705], temp_nodes[141], temp_nodes[140], 208),
      (
          temp_nodes[1870],
          temp_nodes[1608],
          jaxite_bool.constant(False, params),
          4,
      ),
      (temp_nodes[98], temp_nodes[165], jaxite_bool.constant(False, params), 4),
  ]
  outputs = jaxite_bool.pmap_lut3(inputs, sks, params)
  temp_nodes[101] = outputs[0]
  temp_nodes[139] = outputs[1]
  temp_nodes[163] = outputs[2]
  temp_nodes[164] = outputs[3]
  inputs = [
      (temp_nodes[186], my_string[84], jaxite_bool.constant(False, params), 8),
      (temp_nodes[1684], temp_nodes[207], my_string[83], 128),
      (
          temp_nodes[215],
          temp_nodes[1796],
          jaxite_bool.constant(False, params),
          8,
      ),
      (
          temp_nodes[1955],
          temp_nodes[1608],
          jaxite_bool.constant(False, params),
          4,
      ),
  ]
  outputs = jaxite_bool.pmap_lut3(inputs, sks, params)
  temp_nodes[185] = outputs[0]
  temp_nodes[206] = outputs[1]
  temp_nodes[214] = outputs[2]
  temp_nodes[234] = outputs[3]
  inputs = [
      (
          temp_nodes[239],
          temp_nodes[250],
          jaxite_bool.constant(False, params),
          8,
      ),
      (temp_nodes[265], temp_nodes[268], temp_nodes[253], 16),
      (
          temp_nodes[102],
          temp_nodes[1694],
          jaxite_bool.constant(False, params),
          4,
      ),
      (
          temp_nodes[604],
          temp_nodes[274],
          jaxite_bool.constant(False, params),
          8,
      ),
  ]
  outputs = jaxite_bool.pmap_lut3(inputs, sks, params)
  temp_nodes[238] = outputs[0]
  temp_nodes[252] = outputs[1]
  temp_nodes[271] = outputs[2]
  temp_nodes[273] = outputs[3]
  inputs = [
      (temp_nodes[1971], temp_nodes[321], temp_nodes[316], 64),
      (
          temp_nodes[326],
          temp_nodes[1893],
          jaxite_bool.constant(False, params),
          8,
      ),
      (temp_nodes[1958], temp_nodes[1514], temp_nodes[79], 128),
      (temp_nodes[292], temp_nodes[46], jaxite_bool.constant(False, params), 8),
  ]
  outputs = jaxite_bool.pmap_lut3(inputs, sks, params)
  temp_nodes[315] = outputs[0]
  temp_nodes[325] = outputs[1]
  temp_nodes[343] = outputs[2]
  temp_nodes[353] = outputs[3]
  inputs = [
      (temp_nodes[12], temp_nodes[100], temp_nodes[356], 128),
      (temp_nodes[336], temp_nodes[11], jaxite_bool.constant(False, params), 4),
      (temp_nodes[105], temp_nodes[370], temp_nodes[371], 64),
      (temp_nodes[395], temp_nodes[385], temp_nodes[150], 224),
  ]
  outputs = jaxite_bool.pmap_lut3(inputs, sks, params)
  temp_nodes[355] = outputs[0]
  temp_nodes[366] = outputs[1]
  temp_nodes[369] = outputs[2]
  temp_nodes[384] = outputs[3]
  inputs = [
      (temp_nodes[405], temp_nodes[259], temp_nodes[260], 16),
      (temp_nodes[411], my_string[207], temp_nodes[407], 16),
      (
          temp_nodes[402],
          temp_nodes[400],
          jaxite_bool.constant(False, params),
          8,
      ),
      (
          temp_nodes[416],
          temp_nodes[150],
          jaxite_bool.constant(False, params),
          8,
      ),
  ]
  outputs = jaxite_bool.pmap_lut3(inputs, sks, params)
  temp_nodes[404] = outputs[0]
  temp_nodes[406] = outputs[1]
  temp_nodes[414] = outputs[2]
  temp_nodes[415] = outputs[3]
  inputs = [
      (temp_nodes[107], temp_nodes[71], jaxite_bool.constant(False, params), 8),
      (my_string[55], temp_nodes[430], temp_nodes[80], 64),
      (temp_nodes[150], temp_nodes[396], temp_nodes[254], 44),
      (
          temp_nodes[451],
          temp_nodes[452],
          jaxite_bool.constant(False, params),
          8,
      ),
  ]
  outputs = jaxite_bool.pmap_lut3(inputs, sks, params)
  temp_nodes[422] = outputs[0]
  temp_nodes[429] = outputs[1]
  temp_nodes[449] = outputs[2]
  temp_nodes[450] = outputs[3]
  inputs = [
      (
          temp_nodes[149],
          temp_nodes[453],
          jaxite_bool.constant(False, params),
          4,
      ),
      (temp_nodes[461], my_string[214], jaxite_bool.constant(False, params), 1),
      (
          temp_nodes[467],
          temp_nodes[260],
          jaxite_bool.constant(False, params),
          8,
      ),
      (temp_nodes[150], temp_nodes[469], temp_nodes[254], 44),
  ]
  outputs = jaxite_bool.pmap_lut3(inputs, sks, params)
  temp_nodes[457] = outputs[0]
  temp_nodes[462] = outputs[1]
  temp_nodes[466] = outputs[2]
  temp_nodes[468] = outputs[3]
  inputs = [
      (my_string[127], temp_nodes[1712], temp_nodes[489], 13),
      (temp_nodes[61], temp_nodes[547], temp_nodes[387], 13),
      (
          temp_nodes[248],
          temp_nodes[255],
          jaxite_bool.constant(False, params),
          8,
      ),
      (my_string[220], my_string[221], jaxite_bool.constant(False, params), 8),
  ]
  outputs = jaxite_bool.pmap_lut3(inputs, sks, params)
  temp_nodes[488] = outputs[0]
  temp_nodes[546] = outputs[1]
  temp_nodes[550] = outputs[2]
  temp_nodes[560] = outputs[3]
  inputs = [
      (temp_nodes[29], temp_nodes[169], jaxite_bool.constant(False, params), 4),
      (my_string[211], temp_nodes[1532], my_string[214], 7),
      (my_string[216], my_string[223], my_string[217], 7),
  ]
  outputs = jaxite_bool.pmap_lut3(inputs, sks, params)
  temp_nodes[597] = outputs[0]
  temp_nodes[1533] = outputs[1]
  temp_nodes[1536] = outputs[2]
  inputs = [
      (my_string[73], temp_nodes[91], temp_nodes[1607], 64),
      (temp_nodes[1689], my_string[91], jaxite_bool.constant(False, params), 8),
      (my_string[126], my_string[127], jaxite_bool.constant(False, params), 1),
      (temp_nodes[96], temp_nodes[97], temp_nodes[101], 128),
  ]
  outputs = jaxite_bool.pmap_lut3(inputs, sks, params)
  temp_nodes[1606] = outputs[0]
  temp_nodes[1688] = outputs[1]
  temp_nodes[1710] = outputs[2]
  temp_nodes[95] = outputs[3]
  inputs = [
      (temp_nodes[133], temp_nodes[96], jaxite_bool.constant(False, params), 4),
      (temp_nodes[139], temp_nodes[163], temp_nodes[164], 128),
      (
          temp_nodes[185],
          temp_nodes[206],
          jaxite_bool.constant(False, params),
          4,
      ),
      (
          temp_nodes[211],
          temp_nodes[214],
          jaxite_bool.constant(False, params),
          1,
      ),
  ]
  outputs = jaxite_bool.pmap_lut3(inputs, sks, params)
  temp_nodes[137] = outputs[0]
  temp_nodes[138] = outputs[1]
  temp_nodes[184] = outputs[2]
  temp_nodes[210] = outputs[3]
  inputs = [
      (temp_nodes[219], temp_nodes[198], my_string[88], 128),
      (temp_nodes[1607], my_string[76], jaxite_bool.constant(False, params), 8),
      (
          temp_nodes[234],
          temp_nodes[1853],
          jaxite_bool.constant(False, params),
          8,
      ),
      (temp_nodes[252], temp_nodes[238], my_string[124], 64),
  ]
  outputs = jaxite_bool.pmap_lut3(inputs, sks, params)
  temp_nodes[218] = outputs[0]
  temp_nodes[227] = outputs[1]
  temp_nodes[233] = outputs[2]
  temp_nodes[237] = outputs[3]
  inputs = [
      (temp_nodes[98], temp_nodes[99], jaxite_bool.constant(False, params), 4),
      (temp_nodes[133], temp_nodes[1609], temp_nodes[273], 64),
      (temp_nodes[1486], temp_nodes[79], temp_nodes[86], 64),
      (temp_nodes[315], temp_nodes[1931], temp_nodes[83], 128),
  ]
  outputs = jaxite_bool.pmap_lut3(inputs, sks, params)
  temp_nodes[270] = outputs[0]
  temp_nodes[272] = outputs[1]
  temp_nodes[313] = outputs[2]
  temp_nodes[314] = outputs[3]
  inputs = [
      (temp_nodes[105], temp_nodes[11], temp_nodes[343], 64),
      (temp_nodes[49], temp_nodes[353], temp_nodes[73], 64),
      (temp_nodes[325], temp_nodes[355], temp_nodes[99], 64),
      (temp_nodes[89], temp_nodes[366], temp_nodes[1609], 64),
  ]
  outputs = jaxite_bool.pmap_lut3(inputs, sks, params)
  temp_nodes[342] = outputs[0]
  temp_nodes[352] = outputs[1]
  temp_nodes[354] = outputs[2]
  temp_nodes[365] = outputs[3]
  inputs = [
      (temp_nodes[271], temp_nodes[96], temp_nodes[369], 64),
      (temp_nodes[254], temp_nodes[396], temp_nodes[384], 7),
      (
          temp_nodes[402],
          temp_nodes[400],
          jaxite_bool.constant(False, params),
          6,
      ),
      (
          temp_nodes[406],
          temp_nodes[404],
          jaxite_bool.constant(False, params),
          4,
      ),
  ]
  outputs = jaxite_bool.pmap_lut3(inputs, sks, params)
  temp_nodes[368] = outputs[0]
  temp_nodes[383] = outputs[1]
  temp_nodes[401] = outputs[2]
  temp_nodes[403] = outputs[3]
  inputs = [
      (temp_nodes[415], temp_nodes[160], temp_nodes[414], 58),
      (temp_nodes[268], temp_nodes[260], temp_nodes[259], 16),
      (temp_nodes[143], temp_nodes[1868], temp_nodes[422], 16),
      (temp_nodes[1490], temp_nodes[1488], temp_nodes[1972], 16),
  ]
  outputs = jaxite_bool.pmap_lut3(inputs, sks, params)
  temp_nodes[413] = outputs[0]
  temp_nodes[418] = outputs[1]
  temp_nodes[421] = outputs[2]
  temp_nodes[426] = outputs[3]
  inputs = [
      (temp_nodes[179], my_string[119], temp_nodes[429], 16),
      (
          temp_nodes[175],
          temp_nodes[1914],
          jaxite_bool.constant(False, params),
          4,
      ),
      (
          temp_nodes[449],
          temp_nodes[450],
          jaxite_bool.constant(False, params),
          8,
      ),
      (
          temp_nodes[457],
          temp_nodes[404],
          jaxite_bool.constant(False, params),
          1,
      ),
  ]
  outputs = jaxite_bool.pmap_lut3(inputs, sks, params)
  temp_nodes[428] = outputs[0]
  temp_nodes[431] = outputs[1]
  temp_nodes[448] = outputs[2]
  temp_nodes[456] = outputs[3]
  inputs = [
      (temp_nodes[260], temp_nodes[259], temp_nodes[406], 208),
      (temp_nodes[462], my_string[215], temp_nodes[1533], 16),
      (
          temp_nodes[466],
          temp_nodes[452],
          jaxite_bool.constant(False, params),
          8,
      ),
      (
          temp_nodes[468],
          temp_nodes[260],
          jaxite_bool.constant(False, params),
          8,
      ),
  ]
  outputs = jaxite_bool.pmap_lut3(inputs, sks, params)
  temp_nodes[458] = outputs[0]
  temp_nodes[459] = outputs[1]
  temp_nodes[465] = outputs[2]
  temp_nodes[471] = outputs[3]
  inputs = [
      (
          temp_nodes[214],
          temp_nodes[200],
          jaxite_bool.constant(False, params),
          1,
      ),
      (
          temp_nodes[129],
          temp_nodes[488],
          jaxite_bool.constant(False, params),
          8,
      ),
      (temp_nodes[1976], my_string[143], my_string[119], 1),
      (
          temp_nodes[1712],
          my_string[127],
          jaxite_bool.constant(False, params),
          8,
      ),
  ]
  outputs = jaxite_bool.pmap_lut3(inputs, sks, params)
  temp_nodes[477] = outputs[0]
  temp_nodes[487] = outputs[1]
  temp_nodes[490] = outputs[2]
  temp_nodes[492] = outputs[3]
  inputs = [
      (
          temp_nodes[1730],
          my_string[135],
          jaxite_bool.constant(False, params),
          8,
      ),
      (
          temp_nodes[469],
          temp_nodes[254],
          jaxite_bool.constant(False, params),
          8,
      ),
      (temp_nodes[115], temp_nodes[546], temp_nodes[550], 13),
      (temp_nodes[597], temp_nodes[1936], temp_nodes[180], 128),
  ]
  outputs = jaxite_bool.pmap_lut3(inputs, sks, params)
  temp_nodes[493] = outputs[0]
  temp_nodes[544] = outputs[1]
  temp_nodes[545] = outputs[2]
  temp_nodes[596] = outputs[3]
  inputs = [
      (my_string[218], temp_nodes[1536], temp_nodes[560], 176),
  ]
  outputs = jaxite_bool.pmap_lut3(inputs, sks, params)
  temp_nodes[1537] = outputs[0]
  inputs = [
      (
          temp_nodes[1606],
          temp_nodes[95],
          jaxite_bool.constant(False, params),
          8,
      ),
      (temp_nodes[137], temp_nodes[138], temp_nodes[184], 128),
      (temp_nodes[210], temp_nodes[199], temp_nodes[218], 128),
      (
          temp_nodes[1614],
          temp_nodes[1828],
          jaxite_bool.constant(False, params),
          8,
      ),
  ]
  outputs = jaxite_bool.pmap_lut3(inputs, sks, params)
  temp_nodes[1605] = outputs[0]
  temp_nodes[136] = outputs[1]
  temp_nodes[209] = outputs[2]
  temp_nodes[220] = outputs[3]
  inputs = [
      (temp_nodes[1688], my_string[90], my_string[93], 128),
      (temp_nodes[52], temp_nodes[124], temp_nodes[1822], 64),
      (temp_nodes[227], my_string[77], jaxite_bool.constant(False, params), 8),
      (temp_nodes[95], temp_nodes[1606], temp_nodes[233], 112),
  ]
  outputs = jaxite_bool.pmap_lut3(inputs, sks, params)
  temp_nodes[221] = outputs[0]
  temp_nodes[223] = outputs[1]
  temp_nodes[226] = outputs[2]
  temp_nodes[232] = outputs[3]
  inputs = [
      (my_string[125], temp_nodes[237], temp_nodes[1710], 112),
      (temp_nodes[271], temp_nodes[270], temp_nodes[272], 64),
      (temp_nodes[1870], temp_nodes[73], temp_nodes[292], 64),
      (temp_nodes[49], temp_nodes[99], temp_nodes[100], 64),
  ]
  outputs = jaxite_bool.pmap_lut3(inputs, sks, params)
  temp_nodes[236] = outputs[0]
  temp_nodes[269] = outputs[1]
  temp_nodes[291] = outputs[2]
  temp_nodes[311] = outputs[3]
  inputs = [
      (temp_nodes[46], temp_nodes[313], temp_nodes[314], 128),
      (temp_nodes[133], temp_nodes[134], temp_nodes[342], 16),
      (temp_nodes[352], temp_nodes[354], temp_nodes[365], 128),
      (temp_nodes[1606], temp_nodes[368], temp_nodes[97], 128),
  ]
  outputs = jaxite_bool.pmap_lut3(inputs, sks, params)
  temp_nodes[312] = outputs[0]
  temp_nodes[341] = outputs[1]
  temp_nodes[351] = outputs[2]
  temp_nodes[367] = outputs[3]
  inputs = [
      (temp_nodes[401], temp_nodes[383], temp_nodes[403], 64),
      (
          temp_nodes[413],
          temp_nodes[260],
          jaxite_bool.constant(False, params),
          8,
      ),
      (temp_nodes[402], temp_nodes[416], temp_nodes[418], 64),
      (temp_nodes[421], my_string[132], jaxite_bool.constant(False, params), 8),
  ]
  outputs = jaxite_bool.pmap_lut3(inputs, sks, params)
  temp_nodes[382] = outputs[0]
  temp_nodes[412] = outputs[1]
  temp_nodes[417] = outputs[2]
  temp_nodes[420] = outputs[3]
  inputs = [
      (temp_nodes[1482], temp_nodes[426], temp_nodes[170], 64),
      (temp_nodes[428], temp_nodes[128], temp_nodes[431], 128),
      (
          temp_nodes[1725],
          temp_nodes[1498],
          jaxite_bool.constant(False, params),
          1,
      ),
      (temp_nodes[260], temp_nodes[383], temp_nodes[448], 13),
  ]
  outputs = jaxite_bool.pmap_lut3(inputs, sks, params)
  temp_nodes[425] = outputs[0]
  temp_nodes[427] = outputs[1]
  temp_nodes[432] = outputs[2]
  temp_nodes[447] = outputs[3]
  inputs = [
      (temp_nodes[451], temp_nodes[452], temp_nodes[406], 96),
      (temp_nodes[459], temp_nodes[458], temp_nodes[456], 64),
      (
          temp_nodes[465],
          temp_nodes[468],
          jaxite_bool.constant(False, params),
          8,
      ),
      (temp_nodes[465], temp_nodes[471], temp_nodes[406], 224),
  ]
  outputs = jaxite_bool.pmap_lut3(inputs, sks, params)
  temp_nodes[454] = outputs[0]
  temp_nodes[455] = outputs[1]
  temp_nodes[464] = outputs[2]
  temp_nodes[470] = outputs[3]
  inputs = [
      (
          temp_nodes[449],
          temp_nodes[260],
          jaxite_bool.constant(False, params),
          8,
      ),
      (temp_nodes[197], temp_nodes[1873], temp_nodes[477], 16),
      (temp_nodes[487], temp_nodes[81], temp_nodes[490], 128),
      (temp_nodes[492], temp_nodes[1922], temp_nodes[493], 1),
  ]
  outputs = jaxite_bool.pmap_lut3(inputs, sks, params)
  temp_nodes[475] = outputs[0]
  temp_nodes[481] = outputs[1]
  temp_nodes[486] = outputs[2]
  temp_nodes[491] = outputs[3]
  inputs = [
      (
          temp_nodes[457],
          temp_nodes[406],
          jaxite_bool.constant(False, params),
          8,
      ),
      (temp_nodes[150], temp_nodes[545], temp_nodes[544], 13),
      (
          temp_nodes[453],
          temp_nodes[406],
          jaxite_bool.constant(False, params),
          4,
      ),
      (temp_nodes[560], my_string[222], jaxite_bool.constant(False, params), 1),
  ]
  outputs = jaxite_bool.pmap_lut3(inputs, sks, params)
  temp_nodes[540] = outputs[0]
  temp_nodes[543] = outputs[1]
  temp_nodes[557] = outputs[2]
  temp_nodes[561] = outputs[3]
  inputs = [
      (
          temp_nodes[458],
          temp_nodes[459],
          jaxite_bool.constant(False, params),
          4,
      ),
      (
          temp_nodes[141],
          temp_nodes[1705],
          jaxite_bool.constant(False, params),
          4,
      ),
      (temp_nodes[134], temp_nodes[89], temp_nodes[596], 16),
      (my_string[226], my_string[225], my_string[227], 224),
  ]
  outputs = jaxite_bool.pmap_lut3(inputs, sks, params)
  temp_nodes[562] = outputs[0]
  temp_nodes[594] = outputs[1]
  temp_nodes[595] = outputs[2]
  temp_nodes[646] = outputs[3]
  inputs = [
      (my_string[219], temp_nodes[1537], my_string[222], 7),
      (temp_nodes[452], temp_nodes[457], temp_nodes[466], 30),
  ]
  outputs = jaxite_bool.pmap_lut3(inputs, sks, params)
  temp_nodes[1538] = outputs[0]
  temp_nodes[1539] = outputs[1]
  inputs = [
      (temp_nodes[1605], temp_nodes[136], temp_nodes[1608], 16),
      (my_string[134], my_string[135], jaxite_bool.constant(False, params), 1),
      (temp_nodes[220], temp_nodes[209], temp_nodes[221], 64),
      (my_string[89], temp_nodes[223], jaxite_bool.constant(False, params), 4),
  ]
  outputs = jaxite_bool.pmap_lut3(inputs, sks, params)
  temp_nodes[1604] = outputs[0]
  temp_nodes[1729] = outputs[1]
  temp_nodes[208] = outputs[2]
  temp_nodes[222] = outputs[3]
  inputs = [
      (temp_nodes[226], temp_nodes[136], temp_nodes[1679], 208),
      (
          temp_nodes[136],
          temp_nodes[141],
          jaxite_bool.constant(False, params),
          4,
      ),
      (temp_nodes[1605], temp_nodes[136], temp_nodes[105], 1),
      (temp_nodes[1683], temp_nodes[232], temp_nodes[1684], 112),
  ]
  outputs = jaxite_bool.pmap_lut3(inputs, sks, params)
  temp_nodes[225] = outputs[0]
  temp_nodes[228] = outputs[1]
  temp_nodes[230] = outputs[2]
  temp_nodes[231] = outputs[3]
  inputs = [
      (temp_nodes[236], temp_nodes[269], temp_nodes[163], 64),
      (temp_nodes[291], temp_nodes[311], temp_nodes[312], 128),
      (temp_nodes[336], temp_nodes[89], temp_nodes[12], 16),
      (temp_nodes[96], temp_nodes[140], jaxite_bool.constant(False, params), 8),
  ]
  outputs = jaxite_bool.pmap_lut3(inputs, sks, params)
  temp_nodes[235] = outputs[0]
  temp_nodes[290] = outputs[1]
  temp_nodes[335] = outputs[2]
  temp_nodes[339] = outputs[3]
  inputs = [
      (temp_nodes[271], temp_nodes[341], temp_nodes[73], 64),
      (
          temp_nodes[351],
          temp_nodes[367],
          jaxite_bool.constant(False, params),
          1,
      ),
      (temp_nodes[412], temp_nodes[382], temp_nodes[417], 7),
      (temp_nodes[420], my_string[133], jaxite_bool.constant(False, params), 8),
  ]
  outputs = jaxite_bool.pmap_lut3(inputs, sks, params)
  temp_nodes[340] = outputs[0]
  temp_nodes[350] = outputs[1]
  temp_nodes[381] = outputs[2]
  temp_nodes[419] = outputs[3]
  inputs = [
      (temp_nodes[425], temp_nodes[427], temp_nodes[432], 128),
      (temp_nodes[454], temp_nodes[447], temp_nodes[455], 64),
      (
          temp_nodes[464],
          temp_nodes[470],
          jaxite_bool.constant(False, params),
          4,
      ),
      (
          temp_nodes[382],
          temp_nodes[475],
          jaxite_bool.constant(False, params),
          8,
      ),
  ]
  outputs = jaxite_bool.pmap_lut3(inputs, sks, params)
  temp_nodes[424] = outputs[0]
  temp_nodes[446] = outputs[1]
  temp_nodes[463] = outputs[2]
  temp_nodes[474] = outputs[3]
  inputs = [
      (temp_nodes[397], temp_nodes[254], temp_nodes[418], 64),
      (temp_nodes[234], temp_nodes[481], temp_nodes[181], 128),
      (temp_nodes[486], temp_nodes[176], temp_nodes[491], 128),
      (
          temp_nodes[466],
          temp_nodes[540],
          jaxite_bool.constant(False, params),
          8,
      ),
  ]
  outputs = jaxite_bool.pmap_lut3(inputs, sks, params)
  temp_nodes[476] = outputs[0]
  temp_nodes[480] = outputs[1]
  temp_nodes[485] = outputs[2]
  temp_nodes[539] = outputs[3]
  inputs = [
      (
          temp_nodes[543],
          temp_nodes[260],
          jaxite_bool.constant(False, params),
          4,
      ),
      (temp_nodes[540], temp_nodes[454], temp_nodes[459], 224),
      (
          temp_nodes[451],
          temp_nodes[540],
          jaxite_bool.constant(False, params),
          8,
      ),
      (temp_nodes[561], my_string[223], temp_nodes[1538], 16),
  ]
  outputs = jaxite_bool.pmap_lut3(inputs, sks, params)
  temp_nodes[542] = outputs[0]
  temp_nodes[554] = outputs[1]
  temp_nodes[555] = outputs[2]
  temp_nodes[558] = outputs[3]
  inputs = [
      (
          temp_nodes[247],
          temp_nodes[244],
          jaxite_bool.constant(False, params),
          1,
      ),
      (
          temp_nodes[367],
          temp_nodes[137],
          jaxite_bool.constant(False, params),
          4,
      ),
      (temp_nodes[594], temp_nodes[140], temp_nodes[595], 64),
      (
          temp_nodes[557],
          temp_nodes[562],
          jaxite_bool.constant(False, params),
          8,
      ),
  ]
  outputs = jaxite_bool.pmap_lut3(inputs, sks, params)
  temp_nodes[566] = outputs[0]
  temp_nodes[592] = outputs[1]
  temp_nodes[593] = outputs[2]
  temp_nodes[637] = outputs[3]
  inputs = [
      (temp_nodes[406], temp_nodes[456], temp_nodes[459], 208),
      (temp_nodes[646], my_string[230], jaxite_bool.constant(False, params), 1),
      (my_string[229], my_string[228], my_string[230], 7),
      (my_string[237], my_string[236], my_string[238], 7),
  ]
  outputs = jaxite_bool.pmap_lut3(inputs, sks, params)
  temp_nodes[642] = outputs[0]
  temp_nodes[645] = outputs[1]
  temp_nodes[647] = outputs[2]
  temp_nodes[704] = outputs[3]
  inputs = [
      (temp_nodes[406], temp_nodes[1539], temp_nodes[459], 128),
  ]
  outputs = jaxite_bool.pmap_lut3(inputs, sks, params)
  temp_nodes[1540] = outputs[0]
  inputs = [
      (temp_nodes[1604], temp_nodes[208], temp_nodes[222], 128),
      (temp_nodes[1705], temp_nodes[228], temp_nodes[225], 13),
      (temp_nodes[231], temp_nodes[230], temp_nodes[235], 64),
      (
          temp_nodes[1605],
          temp_nodes[136],
          jaxite_bool.constant(False, params),
          1,
      ),
  ]
  outputs = jaxite_bool.pmap_lut3(inputs, sks, params)
  temp_nodes[1603] = outputs[0]
  temp_nodes[224] = outputs[1]
  temp_nodes[229] = outputs[2]
  temp_nodes[281] = outputs[3]
  inputs = [
      (temp_nodes[325], temp_nodes[290], temp_nodes[335], 64),
      (temp_nodes[339], temp_nodes[340], temp_nodes[270], 128),
      (temp_nodes[29], temp_nodes[350], jaxite_bool.constant(False, params), 4),
      (temp_nodes[419], temp_nodes[381], temp_nodes[1729], 112),
  ]
  outputs = jaxite_bool.pmap_lut3(inputs, sks, params)
  temp_nodes[289] = outputs[0]
  temp_nodes[338] = outputs[1]
  temp_nodes[349] = outputs[2]
  temp_nodes[380] = outputs[3]
  inputs = [
      (temp_nodes[1606], temp_nodes[163], temp_nodes[424], 128),
      (
          temp_nodes[446],
          temp_nodes[463],
          jaxite_bool.constant(False, params),
          8,
      ),
      (
          temp_nodes[474],
          temp_nodes[476],
          jaxite_bool.constant(False, params),
          1,
      ),
      (
          temp_nodes[1605],
          temp_nodes[480],
          jaxite_bool.constant(False, params),
          4,
      ),
  ]
  outputs = jaxite_bool.pmap_lut3(inputs, sks, params)
  temp_nodes[423] = outputs[0]
  temp_nodes[445] = outputs[1]
  temp_nodes[473] = outputs[2]
  temp_nodes[479] = outputs[3]
  inputs = [
      (
          temp_nodes[485],
          temp_nodes[1972],
          jaxite_bool.constant(False, params),
          8,
      ),
      (
          temp_nodes[1725],
          temp_nodes[1500],
          jaxite_bool.constant(False, params),
          1,
      ),
      (
          temp_nodes[463],
          temp_nodes[539],
          jaxite_bool.constant(False, params),
          8,
      ),
      (temp_nodes[464], temp_nodes[542], temp_nodes[406], 224),
  ]
  outputs = jaxite_bool.pmap_lut3(inputs, sks, params)
  temp_nodes[484] = outputs[0]
  temp_nodes[494] = outputs[1]
  temp_nodes[538] = outputs[2]
  temp_nodes[541] = outputs[3]
  inputs = [
      (temp_nodes[463], temp_nodes[539], temp_nodes[459], 96),
      (
          temp_nodes[555],
          temp_nodes[554],
          jaxite_bool.constant(False, params),
          4,
      ),
      (
          temp_nodes[557],
          temp_nodes[558],
          jaxite_bool.constant(False, params),
          1,
      ),
      (
          temp_nodes[566],
          temp_nodes[210],
          jaxite_bool.constant(False, params),
          8,
      ),
  ]
  outputs = jaxite_bool.pmap_lut3(inputs, sks, params)
  temp_nodes[551] = outputs[0]
  temp_nodes[553] = outputs[1]
  temp_nodes[556] = outputs[2]
  temp_nodes[565] = outputs[3]
  inputs = [
      (
          temp_nodes[1768],
          my_string[143],
          jaxite_bool.constant(False, params),
          4,
      ),
      (
          temp_nodes[1768],
          my_string[143],
          jaxite_bool.constant(False, params),
          8,
      ),
      (temp_nodes[49], temp_nodes[350], jaxite_bool.constant(False, params), 4),
      (temp_nodes[592], temp_nodes[593], temp_nodes[163], 128),
  ]
  outputs = jaxite_bool.pmap_lut3(inputs, sks, params)
  temp_nodes[572] = outputs[0]
  temp_nodes[574] = outputs[1]
  temp_nodes[590] = outputs[2]
  temp_nodes[591] = outputs[3]
  inputs = [
      (
          temp_nodes[1540],
          temp_nodes[637],
          jaxite_bool.constant(False, params),
          8,
      ),
      (temp_nodes[645], temp_nodes[647], my_string[231], 1),
      (
          temp_nodes[642],
          temp_nodes[558],
          jaxite_bool.constant(False, params),
          8,
      ),
      (temp_nodes[704], my_string[239], jaxite_bool.constant(False, params), 1),
  ]
  outputs = jaxite_bool.pmap_lut3(inputs, sks, params)
  temp_nodes[636] = outputs[0]
  temp_nodes[644] = outputs[1]
  temp_nodes[694] = outputs[2]
  temp_nodes[703] = outputs[3]
  inputs = [
      (my_string[233], my_string[234], my_string[235], 224),
      (temp_nodes[450], temp_nodes[475], temp_nodes[406], 224),
      (
          temp_nodes[401],
          temp_nodes[452],
          jaxite_bool.constant(False, params),
          8,
      ),
      (
          temp_nodes[401],
          temp_nodes[260],
          jaxite_bool.constant(False, params),
          8,
      ),
  ]
  outputs = jaxite_bool.pmap_lut3(inputs, sks, params)
  temp_nodes[705] = outputs[0]
  temp_nodes[710] = outputs[1]
  temp_nodes[720] = outputs[2]
  temp_nodes[721] = outputs[3]
  inputs = [
      (temp_nodes[260], temp_nodes[452], temp_nodes[401], 211),
  ]
  outputs = jaxite_bool.pmap_lut3(inputs, sks, params)
  temp_nodes[1541] = outputs[0]
  inputs = [
      (temp_nodes[1603], temp_nodes[224], temp_nodes[229], 128),
      (temp_nodes[277], temp_nodes[278], temp_nodes[279], 128),
      (temp_nodes[51], temp_nodes[69], jaxite_bool.constant(False, params), 8),
      (temp_nodes[289], temp_nodes[1609], temp_nodes[11], 128),
  ]
  outputs = jaxite_bool.pmap_lut3(inputs, sks, params)
  temp_nodes[1602] = outputs[0]
  temp_nodes[276] = outputs[1]
  temp_nodes[282] = outputs[2]
  temp_nodes[288] = outputs[3]
  inputs = [
      (
          temp_nodes[338],
          temp_nodes[1606],
          jaxite_bool.constant(False, params),
          8,
      ),
      (temp_nodes[349], temp_nodes[1946], my_string[61], 128),
      (temp_nodes[380], temp_nodes[231], temp_nodes[423], 16),
      (
          temp_nodes[134],
          temp_nodes[281],
          jaxite_bool.constant(False, params),
          4,
      ),
  ]
  outputs = jaxite_bool.pmap_lut3(inputs, sks, params)
  temp_nodes[337] = outputs[0]
  temp_nodes[348] = outputs[1]
  temp_nodes[379] = outputs[2]
  temp_nodes[433] = outputs[3]
  inputs = [
      (temp_nodes[237], my_string[125], jaxite_bool.constant(False, params), 8),
      (
          temp_nodes[445],
          temp_nodes[247],
          jaxite_bool.constant(False, params),
          1,
      ),
      (temp_nodes[473], temp_nodes[240], temp_nodes[477], 128),
      (temp_nodes[484], temp_nodes[275], temp_nodes[494], 128),
  ]
  outputs = jaxite_bool.pmap_lut3(inputs, sks, params)
  temp_nodes[437] = outputs[0]
  temp_nodes[444] = outputs[1]
  temp_nodes[472] = outputs[2]
  temp_nodes[483] = outputs[3]
  inputs = [
      (temp_nodes[541], temp_nodes[538], temp_nodes[459], 224),
      (temp_nodes[553], temp_nodes[556], temp_nodes[562], 64),
      (
          temp_nodes[474],
          temp_nodes[565],
          jaxite_bool.constant(False, params),
          4,
      ),
      (
          temp_nodes[252],
          temp_nodes[242],
          jaxite_bool.constant(False, params),
          1,
      ),
  ]
  outputs = jaxite_bool.pmap_lut3(inputs, sks, params)
  temp_nodes[537] = outputs[0]
  temp_nodes[552] = outputs[1]
  temp_nodes[564] = outputs[2]
  temp_nodes[567] = outputs[3]
  inputs = [
      (temp_nodes[572], my_string[151], jaxite_bool.constant(False, params), 1),
      (
          temp_nodes[574],
          temp_nodes[493],
          jaxite_bool.constant(False, params),
          1,
      ),
      (
          temp_nodes[590],
          temp_nodes[591],
          jaxite_bool.constant(False, params),
          8,
      ),
      (temp_nodes[185], temp_nodes[479], temp_nodes[206], 64),
  ]
  outputs = jaxite_bool.pmap_lut3(inputs, sks, params)
  temp_nodes[571] = outputs[0]
  temp_nodes[573] = outputs[1]
  temp_nodes[589] = outputs[2]
  temp_nodes[598] = outputs[3]
  inputs = [
      (temp_nodes[106], temp_nodes[367], temp_nodes[1699], 208),
      (temp_nodes[1609], temp_nodes[604], temp_nodes[274], 128),
      (
          temp_nodes[551],
          temp_nodes[636],
          jaxite_bool.constant(False, params),
          8,
      ),
      (temp_nodes[551], temp_nodes[636], temp_nodes[558], 96),
  ]
  outputs = jaxite_bool.pmap_lut3(inputs, sks, params)
  temp_nodes[599] = outputs[0]
  temp_nodes[603] = outputs[1]
  temp_nodes[635] = outputs[2]
  temp_nodes[639] = outputs[3]
  inputs = [
      (temp_nodes[1540], temp_nodes[637], temp_nodes[558], 96),
      (temp_nodes[642], temp_nodes[562], temp_nodes[558], 16),
      (temp_nodes[644], temp_nodes[647], my_string[231], 1),
      (
          temp_nodes[694],
          temp_nodes[458],
          jaxite_bool.constant(False, params),
          8,
      ),
  ]
  outputs = jaxite_bool.pmap_lut3(inputs, sks, params)
  temp_nodes[640] = outputs[0]
  temp_nodes[641] = outputs[1]
  temp_nodes[643] = outputs[2]
  temp_nodes[693] = outputs[3]
  inputs = [
      (
          temp_nodes[458],
          temp_nodes[694],
          jaxite_bool.constant(False, params),
          4,
      ),
      (my_string[238], temp_nodes[705], temp_nodes[703], 224),
      (
          temp_nodes[448],
          temp_nodes[710],
          jaxite_bool.constant(False, params),
          4,
      ),
      (
          temp_nodes[721],
          temp_nodes[540],
          jaxite_bool.constant(False, params),
          8,
      ),
  ]
  outputs = jaxite_bool.pmap_lut3(inputs, sks, params)
  temp_nodes[699] = outputs[0]
  temp_nodes[702] = outputs[1]
  temp_nodes[709] = outputs[2]
  temp_nodes[722] = outputs[3]
  inputs = [
      (temp_nodes[720], temp_nodes[413], temp_nodes[406], 112),
      (
          temp_nodes[231],
          temp_nodes[271],
          jaxite_bool.constant(False, params),
          1,
      ),
      (temp_nodes[457], temp_nodes[1541], temp_nodes[459], 176),
  ]
  outputs = jaxite_bool.pmap_lut3(inputs, sks, params)
  temp_nodes[725] = outputs[0]
  temp_nodes[748] = outputs[1]
  temp_nodes[1542] = outputs[2]
  inputs = [
      (temp_nodes[1602], temp_nodes[98], temp_nodes[281], 16),
      (temp_nodes[1694], my_string[99], jaxite_bool.constant(False, params), 8),
      (my_string[142], my_string[143], jaxite_bool.constant(False, params), 1),
      (temp_nodes[282], my_string[96], jaxite_bool.constant(False, params), 8),
  ]
  outputs = jaxite_bool.pmap_lut3(inputs, sks, params)
  temp_nodes[1601] = outputs[0]
  temp_nodes[1693] = outputs[1]
  temp_nodes[1766] = outputs[2]
  temp_nodes[285] = outputs[3]
  inputs = [
      (
          temp_nodes[288],
          temp_nodes[337],
          jaxite_bool.constant(False, params),
          1,
      ),
      (temp_nodes[348], temp_nodes[1602], temp_nodes[3], 208),
      (temp_nodes[379], temp_nodes[433], temp_nodes[230], 128),
      (temp_nodes[1613], temp_nodes[1604], temp_nodes[1689], 112),
  ]
  outputs = jaxite_bool.pmap_lut3(inputs, sks, params)
  temp_nodes[287] = outputs[0]
  temp_nodes[347] = outputs[1]
  temp_nodes[378] = outputs[2]
  temp_nodes[434] = outputs[3]
  inputs = [
      (
          temp_nodes[1602],
          temp_nodes[437],
          jaxite_bool.constant(False, params),
          4,
      ),
      (temp_nodes[444], temp_nodes[472], temp_nodes[1765], 128),
      (
          temp_nodes[483],
          temp_nodes[180],
          jaxite_bool.constant(False, params),
          8,
      ),
      (
          temp_nodes[1602],
          temp_nodes[281],
          jaxite_bool.constant(False, params),
          4,
      ),
  ]
  outputs = jaxite_bool.pmap_lut3(inputs, sks, params)
  temp_nodes[436] = outputs[0]
  temp_nodes[443] = outputs[1]
  temp_nodes[482] = outputs[2]
  temp_nodes[497] = outputs[3]
  inputs = [
      (temp_nodes[1602], temp_nodes[136], temp_nodes[1608], 16),
      (temp_nodes[380], temp_nodes[231], temp_nodes[1606], 16),
      (
          temp_nodes[433],
          temp_nodes[230],
          jaxite_bool.constant(False, params),
          8,
      ),
      (temp_nodes[537], temp_nodes[551], temp_nodes[552], 64),
  ]
  outputs = jaxite_bool.pmap_lut3(inputs, sks, params)
  temp_nodes[524] = outputs[0]
  temp_nodes[527] = outputs[1]
  temp_nodes[528] = outputs[2]
  temp_nodes[536] = outputs[3]
  inputs = [
      (temp_nodes[564], temp_nodes[567], my_string[148], 128),
      (temp_nodes[1494], temp_nodes[571], temp_nodes[573], 64),
      (
          temp_nodes[175],
          temp_nodes[492],
          jaxite_bool.constant(False, params),
          1,
      ),
      (
          temp_nodes[1502],
          temp_nodes[1500],
          jaxite_bool.constant(False, params),
          1,
      ),
  ]
  outputs = jaxite_bool.pmap_lut3(inputs, sks, params)
  temp_nodes[563] = outputs[0]
  temp_nodes[570] = outputs[1]
  temp_nodes[575] = outputs[2]
  temp_nodes[577] = outputs[3]
  inputs = [
      (my_string[71], temp_nodes[488], temp_nodes[276], 64),
      (temp_nodes[599], temp_nodes[598], temp_nodes[589], 64),
      (temp_nodes[271], temp_nodes[270], temp_nodes[603], 64),
      (temp_nodes[635], temp_nodes[537], temp_nodes[558], 224),
  ]
  outputs = jaxite_bool.pmap_lut3(inputs, sks, params)
  temp_nodes[578] = outputs[0]
  temp_nodes[588] = outputs[1]
  temp_nodes[602] = outputs[2]
  temp_nodes[634] = outputs[3]
  inputs = [
      (temp_nodes[640], temp_nodes[639], temp_nodes[641], 64),
      (
          temp_nodes[542],
          temp_nodes[403],
          jaxite_bool.constant(False, params),
          4,
      ),
      (
          temp_nodes[640],
          temp_nodes[693],
          jaxite_bool.constant(False, params),
          8,
      ),
      (temp_nodes[553], temp_nodes[637], temp_nodes[558], 96),
  ]
  outputs = jaxite_bool.pmap_lut3(inputs, sks, params)
  temp_nodes[638] = outputs[0]
  temp_nodes[649] = outputs[1]
  temp_nodes[692] = outputs[2]
  temp_nodes[696] = outputs[3]
  inputs = [
      (
          temp_nodes[699],
          temp_nodes[641],
          jaxite_bool.constant(False, params),
          1,
      ),
      (temp_nodes[558], temp_nodes[562], temp_nodes[643], 208),
      (
          temp_nodes[702],
          temp_nodes[703],
          jaxite_bool.constant(False, params),
          4,
      ),
      (temp_nodes[709], temp_nodes[555], temp_nodes[459], 96),
  ]
  outputs = jaxite_bool.pmap_lut3(inputs, sks, params)
  temp_nodes[698] = outputs[0]
  temp_nodes[700] = outputs[1]
  temp_nodes[701] = outputs[2]
  temp_nodes[708] = outputs[3]
  inputs = [
      (
          temp_nodes[553],
          temp_nodes[637],
          jaxite_bool.constant(False, params),
          8,
      ),
      (temp_nodes[720], temp_nodes[412], temp_nodes[725], 224),
      (temp_nodes[748], temp_nodes[270], temp_nodes[163], 128),
      (temp_nodes[722], temp_nodes[406], temp_nodes[1542], 64),
  ]
  outputs = jaxite_bool.pmap_lut3(inputs, sks, params)
  temp_nodes[711] = outputs[0]
  temp_nodes[724] = outputs[1]
  temp_nodes[747] = outputs[2]
  temp_nodes[1543] = outputs[3]
  inputs = [
      (temp_nodes[1818], temp_nodes[282], temp_nodes[1601], 64),
      (temp_nodes[1818], temp_nodes[285], temp_nodes[1601], 64),
      (temp_nodes[136], temp_nodes[287], temp_nodes[50], 64),
      (temp_nodes[1693], my_string[98], my_string[101], 128),
  ]
  outputs = jaxite_bool.pmap_lut3(inputs, sks, params)
  temp_nodes[1600] = outputs[0]
  temp_nodes[284] = outputs[1]
  temp_nodes[286] = outputs[2]
  temp_nodes[344] = outputs[3]
  inputs = [
      (temp_nodes[347], temp_nodes[434], temp_nodes[378], 16),
      (
          temp_nodes[436],
          temp_nodes[1710],
          jaxite_bool.constant(False, params),
          4,
      ),
      (temp_nodes[228], temp_nodes[1602], temp_nodes[1705], 208),
      (
          temp_nodes[443],
          temp_nodes[1766],
          jaxite_bool.constant(False, params),
          4,
      ),
  ]
  outputs = jaxite_bool.pmap_lut3(inputs, sks, params)
  temp_nodes[346] = outputs[0]
  temp_nodes[435] = outputs[1]
  temp_nodes[438] = outputs[2]
  temp_nodes[442] = outputs[3]
  inputs = [
      (temp_nodes[479], temp_nodes[184], temp_nodes[482], 128),
      (
          temp_nodes[497],
          temp_nodes[102],
          jaxite_bool.constant(False, params),
          8,
      ),
      (
          temp_nodes[524],
          temp_nodes[1871],
          jaxite_bool.constant(False, params),
          8,
      ),
      (temp_nodes[527], temp_nodes[528], temp_nodes[424], 128),
  ]
  outputs = jaxite_bool.pmap_lut3(inputs, sks, params)
  temp_nodes[478] = outputs[0]
  temp_nodes[496] = outputs[1]
  temp_nodes[523] = outputs[2]
  temp_nodes[526] = outputs[3]
  inputs = [
      (temp_nodes[536], temp_nodes[445], temp_nodes[563], 16),
      (temp_nodes[274], temp_nodes[570], temp_nodes[575], 128),
      (temp_nodes[432], temp_nodes[577], temp_nodes[578], 128),
      (
          temp_nodes[588],
          temp_nodes[367],
          jaxite_bool.constant(False, params),
          1,
      ),
  ]
  outputs = jaxite_bool.pmap_lut3(inputs, sks, params)
  temp_nodes[535] = outputs[0]
  temp_nodes[569] = outputs[1]
  temp_nodes[576] = outputs[2]
  temp_nodes[587] = outputs[3]
  inputs = [
      (temp_nodes[231], temp_nodes[602], temp_nodes[163], 64),
      (temp_nodes[225], temp_nodes[598], temp_nodes[482], 64),
      (temp_nodes[634], temp_nodes[643], temp_nodes[638], 16),
      (temp_nodes[467], temp_nodes[471], temp_nodes[649], 64),
  ]
  outputs = jaxite_bool.pmap_lut3(inputs, sks, params)
  temp_nodes[601] = outputs[0]
  temp_nodes[617] = outputs[1]
  temp_nodes[633] = outputs[2]
  temp_nodes[648] = outputs[3]
  inputs = [
      (
          temp_nodes[572],
          temp_nodes[489],
          jaxite_bool.constant(False, params),
          1,
      ),
      (
          temp_nodes[1795],
          my_string[151],
          jaxite_bool.constant(False, params),
          8,
      ),
      (temp_nodes[692], temp_nodes[639], temp_nodes[634], 7),
      (temp_nodes[696], temp_nodes[693], temp_nodes[643], 96),
  ]
  outputs = jaxite_bool.pmap_lut3(inputs, sks, params)
  temp_nodes[657] = outputs[0]
  temp_nodes[659] = outputs[1]
  temp_nodes[691] = outputs[2]
  temp_nodes[695] = outputs[3]
  inputs = [
      (temp_nodes[701], temp_nodes[700], temp_nodes[698], 64),
      (temp_nodes[708], temp_nodes[711], temp_nodes[558], 96),
      (
          temp_nodes[696],
          temp_nodes[693],
          jaxite_bool.constant(False, params),
          8,
      ),
      (
          temp_nodes[1543],
          temp_nodes[637],
          jaxite_bool.constant(False, params),
          8,
      ),
  ]
  outputs = jaxite_bool.pmap_lut3(inputs, sks, params)
  temp_nodes[697] = outputs[0]
  temp_nodes[707] = outputs[1]
  temp_nodes[712] = outputs[2]
  temp_nodes[719] = outputs[3]
  inputs = [
      (temp_nodes[724], temp_nodes[722], temp_nodes[459], 96),
      (temp_nodes[225], temp_nodes[236], temp_nodes[747], 16),
      (
          temp_nodes[228],
          temp_nodes[1705],
          jaxite_bool.constant(False, params),
          4,
      ),
      (temp_nodes[1543], temp_nodes[637], temp_nodes[558], 96),
  ]
  outputs = jaxite_bool.pmap_lut3(inputs, sks, params)
  temp_nodes[723] = outputs[0]
  temp_nodes[746] = outputs[1]
  temp_nodes[750] = outputs[2]
  temp_nodes[805] = outputs[3]
  inputs = [
      (my_string[243], my_string[244], my_string[245], 128),
  ]
  outputs = jaxite_bool.pmap_lut3(inputs, sks, params)
  temp_nodes[820] = outputs[0]
  inputs = [
      (my_string[97], temp_nodes[1600], jaxite_bool.constant(False, params), 4),
      (
          temp_nodes[1699],
          my_string[107],
          jaxite_bool.constant(False, params),
          8,
      ),
      (my_string[150], my_string[151], jaxite_bool.constant(False, params), 1),
      (temp_nodes[286], temp_nodes[284], temp_nodes[344], 64),
  ]
  outputs = jaxite_bool.pmap_lut3(inputs, sks, params)
  temp_nodes[1599] = outputs[0]
  temp_nodes[1698] = outputs[1]
  temp_nodes[1793] = outputs[2]
  temp_nodes[283] = outputs[3]
  inputs = [
      (temp_nodes[435], temp_nodes[438], temp_nodes[346], 16),
      (temp_nodes[442], temp_nodes[225], temp_nodes[478], 16),
      (temp_nodes[497], temp_nodes[107], temp_nodes[124], 128),
      (my_string[104], my_string[109], jaxite_bool.constant(False, params), 8),
  ]
  outputs = jaxite_bool.pmap_lut3(inputs, sks, params)
  temp_nodes[345] = outputs[0]
  temp_nodes[441] = outputs[1]
  temp_nodes[500] = outputs[2]
  temp_nodes[502] = outputs[3]
  inputs = [
      (temp_nodes[523], my_string[53], temp_nodes[1890], 143),
      (temp_nodes[435], temp_nodes[434], temp_nodes[526], 16),
      (
          temp_nodes[347],
          temp_nodes[438],
          jaxite_bool.constant(False, params),
          1,
      ),
      (temp_nodes[535], my_string[149], jaxite_bool.constant(False, params), 8),
  ]
  outputs = jaxite_bool.pmap_lut3(inputs, sks, params)
  temp_nodes[522] = outputs[0]
  temp_nodes[525] = outputs[1]
  temp_nodes[529] = outputs[2]
  temp_nodes[534] = outputs[3]
  inputs = [
      (temp_nodes[1484], temp_nodes[576], temp_nodes[569], 64),
      (
          temp_nodes[133],
          temp_nodes[587],
          jaxite_bool.constant(False, params),
          4,
      ),
      (temp_nodes[236], temp_nodes[230], temp_nodes[601], 64),
      (
          temp_nodes[1603],
          temp_nodes[224],
          jaxite_bool.constant(False, params),
          8,
      ),
  ]
  outputs = jaxite_bool.pmap_lut3(inputs, sks, params)
  temp_nodes[568] = outputs[0]
  temp_nodes[586] = outputs[1]
  temp_nodes[600] = outputs[2]
  temp_nodes[609] = outputs[3]
  inputs = [
      (
          temp_nodes[250],
          temp_nodes[1872],
          jaxite_bool.constant(False, params),
          8,
      ),
      (temp_nodes[442], temp_nodes[434], temp_nodes[617], 16),
      (temp_nodes[1694], temp_nodes[496], temp_nodes[438], 13),
      (temp_nodes[633], temp_nodes[648], temp_nodes[566], 16),
  ]
  outputs = jaxite_bool.pmap_lut3(inputs, sks, params)
  temp_nodes[612] = outputs[0]
  temp_nodes[616] = outputs[1]
  temp_nodes[618] = outputs[2]
  temp_nodes[632] = outputs[3]
  inputs = [
      (
          temp_nodes[536],
          temp_nodes[445],
          jaxite_bool.constant(False, params),
          1,
      ),
      (temp_nodes[252], temp_nodes[241], my_string[156], 64),
      (
          temp_nodes[431],
          temp_nodes[178],
          jaxite_bool.constant(False, params),
          8,
      ),
      (
          temp_nodes[573],
          temp_nodes[657],
          jaxite_bool.constant(False, params),
          8,
      ),
  ]
  outputs = jaxite_bool.pmap_lut3(inputs, sks, params)
  temp_nodes[650] = outputs[0]
  temp_nodes[651] = outputs[1]
  temp_nodes[655] = outputs[2]
  temp_nodes[656] = outputs[3]
  inputs = [
      (temp_nodes[1928], temp_nodes[659], my_string[79], 1),
      (
          temp_nodes[1795],
          my_string[151],
          jaxite_bool.constant(False, params),
          4,
      ),
      (temp_nodes[695], temp_nodes[691], temp_nodes[697], 64),
      (temp_nodes[707], temp_nodes[712], temp_nodes[643], 96),
  ]
  outputs = jaxite_bool.pmap_lut3(inputs, sks, params)
  temp_nodes[658] = outputs[0]
  temp_nodes[661] = outputs[1]
  temp_nodes[690] = outputs[2]
  temp_nodes[706] = outputs[3]
  inputs = [
      (
          temp_nodes[536],
          temp_nodes[648],
          jaxite_bool.constant(False, params),
          1,
      ),
      (
          temp_nodes[445],
          temp_nodes[567],
          jaxite_bool.constant(False, params),
          4,
      ),
      (temp_nodes[719], temp_nodes[723], temp_nodes[558], 96),
      (temp_nodes[696], temp_nodes[643], temp_nodes[641], 16),
  ]
  outputs = jaxite_bool.pmap_lut3(inputs, sks, params)
  temp_nodes[715] = outputs[0]
  temp_nodes[716] = outputs[1]
  temp_nodes[718] = outputs[2]
  temp_nodes[726] = outputs[3]
  inputs = [
      (temp_nodes[746], temp_nodes[1603], temp_nodes[230], 128),
      (temp_nodes[750], temp_nodes[1609], temp_nodes[273], 64),
      (
          temp_nodes[805],
          temp_nodes[693],
          jaxite_bool.constant(False, params),
          8,
      ),
      (my_string[241], my_string[242], temp_nodes[820], 224),
  ]
  outputs = jaxite_bool.pmap_lut3(inputs, sks, params)
  temp_nodes[745] = outputs[0]
  temp_nodes[749] = outputs[1]
  temp_nodes[804] = outputs[2]
  temp_nodes[819] = outputs[3]
  inputs = [
      (temp_nodes[1599], temp_nodes[283], temp_nodes[345], 128),
      (temp_nodes[347], temp_nodes[434], temp_nodes[441], 16),
      (
          temp_nodes[496],
          temp_nodes[1694],
          jaxite_bool.constant(False, params),
          4,
      ),
      (temp_nodes[500], my_string[108], jaxite_bool.constant(False, params), 8),
  ]
  outputs = jaxite_bool.pmap_lut3(inputs, sks, params)
  temp_nodes[1598] = outputs[0]
  temp_nodes[440] = outputs[1]
  temp_nodes[495] = outputs[2]
  temp_nodes[499] = outputs[3]
  inputs = [
      (my_string[105], my_string[106], temp_nodes[502], 64),
      (
          temp_nodes[1599],
          temp_nodes[283],
          jaxite_bool.constant(False, params),
          8,
      ),
      (temp_nodes[522], temp_nodes[525], temp_nodes[529], 128),
      (temp_nodes[1793], temp_nodes[534], temp_nodes[442], 13),
  ]
  outputs = jaxite_bool.pmap_lut3(inputs, sks, params)
  temp_nodes[501] = outputs[0]
  temp_nodes[520] = outputs[1]
  temp_nodes[521] = outputs[2]
  temp_nodes[533] = outputs[3]
  inputs = [
      (temp_nodes[586], temp_nodes[600], temp_nodes[609], 128),
      (
          temp_nodes[240],
          temp_nodes[612],
          jaxite_bool.constant(False, params),
          8,
      ),
      (temp_nodes[347], temp_nodes[618], temp_nodes[616], 64),
      (
          temp_nodes[1698],
          my_string[109],
          jaxite_bool.constant(False, params),
          8,
      ),
  ]
  outputs = jaxite_bool.pmap_lut3(inputs, sks, params)
  temp_nodes[585] = outputs[0]
  temp_nodes[611] = outputs[1]
  temp_nodes[615] = outputs[2]
  temp_nodes[621] = outputs[3]
  inputs = [
      (temp_nodes[632], temp_nodes[650], temp_nodes[651], 128),
      (temp_nodes[655], temp_nodes[656], temp_nodes[658], 128),
      (temp_nodes[661], my_string[159], jaxite_bool.constant(False, params), 1),
      (
          temp_nodes[1502],
          temp_nodes[1504],
          jaxite_bool.constant(False, params),
          1,
      ),
  ]
  outputs = jaxite_bool.pmap_lut3(inputs, sks, params)
  temp_nodes[631] = outputs[0]
  temp_nodes[654] = outputs[1]
  temp_nodes[660] = outputs[2]
  temp_nodes[663] = outputs[3]
  inputs = [
      (
          temp_nodes[534],
          temp_nodes[1793],
          jaxite_bool.constant(False, params),
          4,
      ),
      (temp_nodes[442], temp_nodes[1603], temp_nodes[568], 64),
      (
          temp_nodes[690],
          temp_nodes[706],
          jaxite_bool.constant(False, params),
          8,
      ),
      (
          temp_nodes[715],
          temp_nodes[716],
          jaxite_bool.constant(False, params),
          8,
      ),
  ]
  outputs = jaxite_bool.pmap_lut3(inputs, sks, params)
  temp_nodes[675] = outputs[0]
  temp_nodes[676] = outputs[1]
  temp_nodes[689] = outputs[2]
  temp_nodes[714] = outputs[3]
  inputs = [
      (temp_nodes[634], temp_nodes[718], temp_nodes[726], 64),
      (
          temp_nodes[1813],
          my_string[159],
          jaxite_bool.constant(False, params),
          4,
      ),
      (
          temp_nodes[1813],
          my_string[159],
          jaxite_bool.constant(False, params),
          8,
      ),
      (my_string[151], my_string[143], temp_nodes[280], 16),
  ]
  outputs = jaxite_bool.pmap_lut3(inputs, sks, params)
  temp_nodes[717] = outputs[0]
  temp_nodes[731] = outputs[1]
  temp_nodes[734] = outputs[2]
  temp_nodes[735] = outputs[3]
  inputs = [
      (temp_nodes[586], temp_nodes[745], temp_nodes[749], 128),
      (temp_nodes[434], temp_nodes[231], temp_nodes[433], 16),
      (temp_nodes[380], temp_nodes[1606], temp_nodes[424], 64),
      (temp_nodes[537], temp_nodes[552], temp_nodes[723], 64),
  ]
  outputs = jaxite_bool.pmap_lut3(inputs, sks, params)
  temp_nodes[744] = outputs[0]
  temp_nodes[751] = outputs[1]
  temp_nodes[753] = outputs[2]
  temp_nodes[770] = outputs[3]
  inputs = [
      (temp_nodes[718], temp_nodes[804], temp_nodes[643], 96),
      (
          temp_nodes[699],
          temp_nodes[643],
          jaxite_bool.constant(False, params),
          8,
      ),
      (temp_nodes[819], my_string[246], jaxite_bool.constant(False, params), 1),
      (my_string[247], my_string[244], my_string[245], 64),
  ]
  outputs = jaxite_bool.pmap_lut3(inputs, sks, params)
  temp_nodes[803] = outputs[0]
  temp_nodes[812] = outputs[1]
  temp_nodes[818] = outputs[2]
  temp_nodes[821] = outputs[3]
  inputs = [
      (temp_nodes[436], temp_nodes[1598], temp_nodes[1710], 208),
      (
          temp_nodes[1705],
          my_string[115],
          jaxite_bool.constant(False, params),
          8,
      ),
      (my_string[158], my_string[159], jaxite_bool.constant(False, params), 1),
      (temp_nodes[495], temp_nodes[438], temp_nodes[440], 16),
  ]
  outputs = jaxite_bool.pmap_lut3(inputs, sks, params)
  temp_nodes[1597] = outputs[0]
  temp_nodes[1704] = outputs[1]
  temp_nodes[1811] = outputs[2]
  temp_nodes[439] = outputs[3]
  inputs = [
      (temp_nodes[499], temp_nodes[1698], temp_nodes[501], 64),
      (temp_nodes[1598], temp_nodes[380], temp_nodes[433], 16),
      (temp_nodes[1598], temp_nodes[1602], temp_nodes[136], 1),
      (
          temp_nodes[1826],
          temp_nodes[142],
          jaxite_bool.constant(False, params),
          4,
      ),
  ]
  outputs = jaxite_bool.pmap_lut3(inputs, sks, params)
  temp_nodes[498] = outputs[0]
  temp_nodes[503] = outputs[1]
  temp_nodes[505] = outputs[2]
  temp_nodes[506] = outputs[3]
  inputs = [
      (temp_nodes[481], my_string[116], jaxite_bool.constant(False, params), 8),
      (temp_nodes[1598], temp_nodes[1602], temp_nodes[232], 16),
      (
          temp_nodes[1598],
          temp_nodes[500],
          jaxite_bool.constant(False, params),
          4,
      ),
      (
          temp_nodes[520],
          temp_nodes[521],
          jaxite_bool.constant(False, params),
          8,
      ),
  ]
  outputs = jaxite_bool.pmap_lut3(inputs, sks, params)
  temp_nodes[510] = outputs[0]
  temp_nodes[514] = outputs[1]
  temp_nodes[516] = outputs[2]
  temp_nodes[519] = outputs[3]
  inputs = [
      (
          temp_nodes[1598],
          temp_nodes[380],
          jaxite_bool.constant(False, params),
          1,
      ),
      (temp_nodes[495], temp_nodes[533], temp_nodes[1603], 64),
      (
          temp_nodes[585],
          temp_nodes[587],
          jaxite_bool.constant(False, params),
          4,
      ),
      (temp_nodes[521], temp_nodes[520], temp_nodes[611], 112),
  ]
  outputs = jaxite_bool.pmap_lut3(inputs, sks, params)
  temp_nodes[531] = outputs[0]
  temp_nodes[532] = outputs[1]
  temp_nodes[584] = outputs[2]
  temp_nodes[610] = outputs[3]
  inputs = [
      (temp_nodes[1598], temp_nodes[615], temp_nodes[433], 64),
      (temp_nodes[499], temp_nodes[621], my_string[106], 64),
      (temp_nodes[631], my_string[157], jaxite_bool.constant(False, params), 8),
      (temp_nodes[1490], temp_nodes[654], temp_nodes[660], 64),
  ]
  outputs = jaxite_bool.pmap_lut3(inputs, sks, params)
  temp_nodes[614] = outputs[0]
  temp_nodes[620] = outputs[1]
  temp_nodes[630] = outputs[2]
  temp_nodes[653] = outputs[3]
  inputs = [
      (temp_nodes[1908], temp_nodes[170], temp_nodes[663], 128),
      (temp_nodes[675], temp_nodes[495], temp_nodes[676], 16),
      (
          temp_nodes[689],
          temp_nodes[566],
          jaxite_bool.constant(False, params),
          4,
      ),
      (temp_nodes[717], temp_nodes[714], my_string[164], 64),
  ]
  outputs = jaxite_bool.pmap_lut3(inputs, sks, params)
  temp_nodes[662] = outputs[0]
  temp_nodes[674] = outputs[1]
  temp_nodes[688] = outputs[2]
  temp_nodes[713] = outputs[3]
  inputs = [
      (temp_nodes[1504], temp_nodes[731], my_string[167], 1),
      (
          temp_nodes[734],
          temp_nodes[735],
          jaxite_bool.constant(False, params),
          4,
      ),
      (temp_nodes[744], temp_nodes[751], temp_nodes[230], 64),
      (temp_nodes[435], temp_nodes[347], temp_nodes[753], 16),
  ]
  outputs = jaxite_bool.pmap_lut3(inputs, sks, params)
  temp_nodes[730] = outputs[0]
  temp_nodes[733] = outputs[1]
  temp_nodes[743] = outputs[2]
  temp_nodes[752] = outputs[3]
  inputs = [
      (
          temp_nodes[446],
          temp_nodes[724],
          jaxite_bool.constant(False, params),
          8,
      ),
      (temp_nodes[412], temp_nodes[382], temp_nodes[770], 7),
      (
          temp_nodes[690],
          temp_nodes[803],
          jaxite_bool.constant(False, params),
          8,
      ),
      (
          temp_nodes[691],
          temp_nodes[643],
          jaxite_bool.constant(False, params),
          4,
      ),
  ]
  outputs = jaxite_bool.pmap_lut3(inputs, sks, params)
  temp_nodes[762] = outputs[0]
  temp_nodes[769] = outputs[1]
  temp_nodes[802] = outputs[2]
  temp_nodes[810] = outputs[3]
  inputs = [
      (
          temp_nodes[805],
          temp_nodes[812],
          jaxite_bool.constant(False, params),
          8,
      ),
      (
          temp_nodes[694],
          temp_nodes[643],
          jaxite_bool.constant(False, params),
          4,
      ),
      (
          temp_nodes[700],
          temp_nodes[701],
          jaxite_bool.constant(False, params),
          4,
      ),
      (
          temp_nodes[818],
          temp_nodes[821],
          jaxite_bool.constant(False, params),
          8,
      ),
  ]
  outputs = jaxite_bool.pmap_lut3(inputs, sks, params)
  temp_nodes[811] = outputs[0]
  temp_nodes[815] = outputs[1]
  temp_nodes[816] = outputs[2]
  temp_nodes[817] = outputs[3]
  inputs = [
      (my_string[255], my_string[252], my_string[253], 64),
      (
          temp_nodes[709],
          temp_nodes[555],
          jaxite_bool.constant(False, params),
          8,
      ),
      (temp_nodes[693], temp_nodes[699], temp_nodes[805], 30),
  ]
  outputs = jaxite_bool.pmap_lut3(inputs, sks, params)
  temp_nodes[891] = outputs[0]
  temp_nodes[903] = outputs[1]
  temp_nodes[1544] = outputs[2]
  inputs = [
      (temp_nodes[1597], temp_nodes[439], temp_nodes[498], 64),
      (my_string[166], my_string[167], jaxite_bool.constant(False, params), 1),
      (temp_nodes[505], temp_nodes[506], temp_nodes[162], 128),
      (temp_nodes[505], temp_nodes[506], temp_nodes[510], 128),
  ]
  outputs = jaxite_bool.pmap_lut3(inputs, sks, params)
  temp_nodes[1596] = outputs[0]
  temp_nodes[1841] = outputs[1]
  temp_nodes[504] = outputs[2]
  temp_nodes[509] = outputs[3]
  inputs = [
      (
          temp_nodes[1704],
          my_string[117],
          jaxite_bool.constant(False, params),
          8,
      ),
      (temp_nodes[1683], temp_nodes[514], temp_nodes[1684], 112),
      (temp_nodes[1700], temp_nodes[516], temp_nodes[1699], 112),
      (
          temp_nodes[519],
          temp_nodes[433],
          jaxite_bool.constant(False, params),
          4,
      ),
  ]
  outputs = jaxite_bool.pmap_lut3(inputs, sks, params)
  temp_nodes[511] = outputs[0]
  temp_nodes[513] = outputs[1]
  temp_nodes[515] = outputs[2]
  temp_nodes[518] = outputs[3]
  inputs = [
      (temp_nodes[531], temp_nodes[532], temp_nodes[568], 128),
      (my_string[105], temp_nodes[610], temp_nodes[584], 64),
      (temp_nodes[1597], temp_nodes[531], temp_nodes[614], 64),
      (temp_nodes[620], my_string[104], jaxite_bool.constant(False, params), 8),
  ]
  outputs = jaxite_bool.pmap_lut3(inputs, sks, params)
  temp_nodes[530] = outputs[0]
  temp_nodes[583] = outputs[1]
  temp_nodes[613] = outputs[2]
  temp_nodes[619] = outputs[3]
  inputs = [
      (
          temp_nodes[630],
          temp_nodes[1811],
          jaxite_bool.constant(False, params),
          4,
      ),
      (temp_nodes[653], temp_nodes[662], temp_nodes[494], 128),
      (temp_nodes[1597], temp_nodes[503], temp_nodes[674], 64),
      (temp_nodes[688], temp_nodes[713], my_string[165], 128),
  ]
  outputs = jaxite_bool.pmap_lut3(inputs, sks, params)
  temp_nodes[629] = outputs[0]
  temp_nodes[652] = outputs[1]
  temp_nodes[673] = outputs[2]
  temp_nodes[687] = outputs[3]
  inputs = [
      (
          temp_nodes[275],
          temp_nodes[577],
          jaxite_bool.constant(False, params),
          8,
      ),
      (
          temp_nodes[1506],
          temp_nodes[730],
          jaxite_bool.constant(False, params),
          4,
      ),
      (temp_nodes[79], temp_nodes[132], temp_nodes[733], 128),
      (temp_nodes[743], temp_nodes[522], temp_nodes[752], 128),
  ]
  outputs = jaxite_bool.pmap_lut3(inputs, sks, params)
  temp_nodes[728] = outputs[0]
  temp_nodes[729] = outputs[1]
  temp_nodes[732] = outputs[2]
  temp_nodes[742] = outputs[3]
  inputs = [
      (temp_nodes[802], temp_nodes[143], temp_nodes[108], 1),
      (temp_nodes[762], temp_nodes[417], temp_nodes[769], 16),
      (temp_nodes[811], temp_nodes[803], temp_nodes[810], 7),
      (temp_nodes[815], temp_nodes[817], temp_nodes[816], 16),
  ]
  outputs = jaxite_bool.pmap_lut3(inputs, sks, params)
  temp_nodes[801] = outputs[0]
  temp_nodes[806] = outputs[1]
  temp_nodes[809] = outputs[2]
  temp_nodes[814] = outputs[3]
  inputs = [
      (
          temp_nodes[815],
          temp_nodes[816],
          jaxite_bool.constant(False, params),
          8,
      ),
      (my_string[248], temp_nodes[891], my_string[249], 13),
      (temp_nodes[406], temp_nodes[447], temp_nodes[903], 13),
      (temp_nodes[643], temp_nodes[1544], temp_nodes[701], 128),
  ]
  outputs = jaxite_bool.pmap_lut3(inputs, sks, params)
  temp_nodes[884] = outputs[0]
  temp_nodes[890] = outputs[1]
  temp_nodes[902] = outputs[2]
  temp_nodes[1545] = outputs[3]
  inputs = [
      (
          temp_nodes[1596],
          temp_nodes[503],
          jaxite_bool.constant(False, params),
          8,
      ),
      (temp_nodes[509], temp_nodes[511], my_string[114], 64),
      (temp_nodes[513], temp_nodes[515], temp_nodes[1597], 1),
      (
          temp_nodes[518],
          temp_nodes[530],
          jaxite_bool.constant(False, params),
          8,
      ),
  ]
  outputs = jaxite_bool.pmap_lut3(inputs, sks, params)
  temp_nodes[1595] = outputs[0]
  temp_nodes[508] = outputs[1]
  temp_nodes[512] = outputs[2]
  temp_nodes[517] = outputs[3]
  inputs = [
      (temp_nodes[504], my_string[112], jaxite_bool.constant(False, params), 8),
      (temp_nodes[1602], temp_nodes[136], temp_nodes[226], 16),
      (temp_nodes[583], temp_nodes[613], temp_nodes[619], 128),
      (temp_nodes[503], temp_nodes[1596], temp_nodes[509], 112),
  ]
  outputs = jaxite_bool.pmap_lut3(inputs, sks, params)
  temp_nodes[579] = outputs[0]
  temp_nodes[581] = outputs[1]
  temp_nodes[582] = outputs[2]
  temp_nodes[626] = outputs[3]
  inputs = [
      (temp_nodes[629], temp_nodes[520], temp_nodes[652], 64),
      (my_string[122], my_string[120], my_string[125], 128),
      (temp_nodes[513], temp_nodes[515], temp_nodes[673], 16),
      (
          temp_nodes[687],
          temp_nodes[1841],
          jaxite_bool.constant(False, params),
          4,
      ),
  ]
  outputs = jaxite_bool.pmap_lut3(inputs, sks, params)
  temp_nodes[628] = outputs[0]
  temp_nodes[668] = outputs[1]
  temp_nodes[672] = outputs[2]
  temp_nodes[686] = outputs[3]
  inputs = [
      (temp_nodes[728], temp_nodes[729], temp_nodes[732], 128),
      (temp_nodes[438], temp_nodes[520], temp_nodes[742], 64),
      (
          temp_nodes[801],
          temp_nodes[806],
          jaxite_bool.constant(False, params),
          8,
      ),
      (
          temp_nodes[809],
          temp_nodes[701],
          jaxite_bool.constant(False, params),
          4,
      ),
  ]
  outputs = jaxite_bool.pmap_lut3(inputs, sks, params)
  temp_nodes[727] = outputs[0]
  temp_nodes[741] = outputs[1]
  temp_nodes[800] = outputs[2]
  temp_nodes[808] = outputs[3]
  inputs = [
      (
          temp_nodes[1545],
          temp_nodes[814],
          jaxite_bool.constant(False, params),
          4,
      ),
      (temp_nodes[803], temp_nodes[811], temp_nodes[701], 96),
      (temp_nodes[659], my_string[175], my_string[167], 1),
      (
          temp_nodes[1545],
          temp_nodes[884],
          jaxite_bool.constant(False, params),
          8,
      ),
  ]
  outputs = jaxite_bool.pmap_lut3(inputs, sks, params)
  temp_nodes[813] = outputs[0]
  temp_nodes[822] = outputs[1]
  temp_nodes[832] = outputs[2]
  temp_nodes[883] = outputs[3]
  inputs = [
      (my_string[250], temp_nodes[890], my_string[251], 176),
      (
          temp_nodes[696],
          temp_nodes[812],
          jaxite_bool.constant(False, params),
          8,
      ),
      (
          temp_nodes[902],
          temp_nodes[459],
          jaxite_bool.constant(False, params),
          4,
      ),
  ]
  outputs = jaxite_bool.pmap_lut3(inputs, sks, params)
  temp_nodes[889] = outputs[0]
  temp_nodes[898] = outputs[1]
  temp_nodes[901] = outputs[2]
  inputs = [
      (temp_nodes[1595], my_string[113], temp_nodes[504], 16),
      (temp_nodes[508], temp_nodes[512], temp_nodes[517], 128),
      (temp_nodes[581], temp_nodes[1595], temp_nodes[1679], 208),
      (my_string[117], temp_nodes[626], temp_nodes[1705], 112),
  ]
  outputs = jaxite_bool.pmap_lut3(inputs, sks, params)
  temp_nodes[1594] = outputs[0]
  temp_nodes[507] = outputs[1]
  temp_nodes[580] = outputs[2]
  temp_nodes[625] = outputs[3]
  inputs = [
      (temp_nodes[515], temp_nodes[628], temp_nodes[531], 64),
      (temp_nodes[582], temp_nodes[519], temp_nodes[434], 254),
      (temp_nodes[252], my_string[121], temp_nodes[238], 16),
      (temp_nodes[1710], temp_nodes[668], my_string[123], 128),
  ]
  outputs = jaxite_bool.pmap_lut3(inputs, sks, params)
  temp_nodes[627] = outputs[0]
  temp_nodes[664] = outputs[1]
  temp_nodes[666] = outputs[2]
  temp_nodes[667] = outputs[3]
  inputs = [
      (temp_nodes[508], temp_nodes[579], temp_nodes[672], 128),
      (temp_nodes[686], temp_nodes[629], temp_nodes[727], 16),
      (
          temp_nodes[741],
          temp_nodes[744],
          jaxite_bool.constant(False, params),
          1,
      ),
      (temp_nodes[629], temp_nodes[520], temp_nodes[652], 64),
  ]
  outputs = jaxite_bool.pmap_lut3(inputs, sks, params)
  temp_nodes[671] = outputs[0]
  temp_nodes[685] = outputs[1]
  temp_nodes[740] = outputs[2]
  temp_nodes[787] = outputs[3]
  inputs = [
      (temp_nodes[717], temp_nodes[800], my_string[172], 64),
      (temp_nodes[808], temp_nodes[813], temp_nodes[822], 64),
      (temp_nodes[179], temp_nodes[82], jaxite_bool.constant(False, params), 1),
      (temp_nodes[175], temp_nodes[660], temp_nodes[430], 64),
  ]
  outputs = jaxite_bool.pmap_lut3(inputs, sks, params)
  temp_nodes[799] = outputs[0]
  temp_nodes[807] = outputs[1]
  temp_nodes[829] = outputs[2]
  temp_nodes[830] = outputs[3]
  inputs = [
      (temp_nodes[84], my_string[95], temp_nodes[832], 16),
      (
          temp_nodes[686],
          temp_nodes[727],
          jaxite_bool.constant(False, params),
          4,
      ),
      (
          temp_nodes[583],
          temp_nodes[619],
          jaxite_bool.constant(False, params),
          8,
      ),
      (temp_nodes[822], temp_nodes[883], temp_nodes[808], 7),
  ]
  outputs = jaxite_bool.pmap_lut3(inputs, sks, params)
  temp_nodes[831] = outputs[0]
  temp_nodes[845] = outputs[1]
  temp_nodes[846] = outputs[2]
  temp_nodes[882] = outputs[3]
  inputs = [
      (temp_nodes[643], temp_nodes[698], temp_nodes[701], 208),
      (
          temp_nodes[816],
          temp_nodes[817],
          jaxite_bool.constant(False, params),
          4,
      ),
      (temp_nodes[889], my_string[254], temp_nodes[891], 16),
      (temp_nodes[706], temp_nodes[898], temp_nodes[701], 96),
  ]
  outputs = jaxite_bool.pmap_lut3(inputs, sks, params)
  temp_nodes[886] = outputs[0]
  temp_nodes[887] = outputs[1]
  temp_nodes[888] = outputs[2]
  temp_nodes[897] = outputs[3]
  inputs = [
      (temp_nodes[901], temp_nodes[708], temp_nodes[552], 64),
  ]
  outputs = jaxite_bool.pmap_lut3(inputs, sks, params)
  temp_nodes[900] = outputs[0]
  inputs = [
      (temp_nodes[1594], temp_nodes[507], temp_nodes[579], 128),
      (temp_nodes[580], temp_nodes[625], temp_nodes[627], 16),
      (temp_nodes[237], temp_nodes[666], temp_nodes[667], 64),
      (temp_nodes[580], temp_nodes[1594], temp_nodes[671], 64),
  ]
  outputs = jaxite_bool.pmap_lut3(inputs, sks, params)
  temp_nodes[1593] = outputs[0]
  temp_nodes[624] = outputs[1]
  temp_nodes[665] = outputs[2]
  temp_nodes[670] = outputs[3]
  inputs = [
      (temp_nodes[664], temp_nodes[685], temp_nodes[498], 64),
      (
          temp_nodes[740],
          temp_nodes[437],
          jaxite_bool.constant(False, params),
          8,
      ),
      (
          temp_nodes[580],
          temp_nodes[664],
          jaxite_bool.constant(False, params),
          1,
      ),
      (temp_nodes[515], temp_nodes[787], temp_nodes[531], 64),
  ]
  outputs = jaxite_bool.pmap_lut3(inputs, sks, params)
  temp_nodes[684] = outputs[0]
  temp_nodes[739] = outputs[1]
  temp_nodes[783] = outputs[2]
  temp_nodes[786] = outputs[3]
  inputs = [
      (
          temp_nodes[582],
          temp_nodes[519],
          jaxite_bool.constant(False, params),
          1,
      ),
      (
          temp_nodes[807],
          temp_nodes[799],
          jaxite_bool.constant(False, params),
          4,
      ),
      (
          temp_nodes[1494],
          temp_nodes[1492],
          jaxite_bool.constant(False, params),
          1,
      ),
      (
          temp_nodes[1506],
          temp_nodes[1508],
          jaxite_bool.constant(False, params),
          1,
      ),
  ]
  outputs = jaxite_bool.pmap_lut3(inputs, sks, params)
  temp_nodes[793] = outputs[0]
  temp_nodes[798] = outputs[1]
  temp_nodes[825] = outputs[2]
  temp_nodes[826] = outputs[3]
  inputs = [
      (temp_nodes[829], temp_nodes[830], temp_nodes[831], 128),
      (temp_nodes[664], temp_nodes[845], temp_nodes[846], 64),
      (
          temp_nodes[882],
          temp_nodes[817],
          jaxite_bool.constant(False, params),
          4,
      ),
      (temp_nodes[886], temp_nodes[888], temp_nodes[887], 16),
  ]
  outputs = jaxite_bool.pmap_lut3(inputs, sks, params)
  temp_nodes[828] = outputs[0]
  temp_nodes[844] = outputs[1]
  temp_nodes[881] = outputs[2]
  temp_nodes[885] = outputs[3]
  inputs = [
      (temp_nodes[1545], temp_nodes[884], temp_nodes[817], 144),
      (temp_nodes[808], temp_nodes[813], temp_nodes[897], 64),
      (temp_nodes[689], temp_nodes[900], temp_nodes[444], 16),
      (
          temp_nodes[1843],
          my_string[167],
          jaxite_bool.constant(False, params),
          8,
      ),
  ]
  outputs = jaxite_bool.pmap_lut3(inputs, sks, params)
  temp_nodes[893] = outputs[0]
  temp_nodes[896] = outputs[1]
  temp_nodes[899] = outputs[2]
  temp_nodes[910] = outputs[3]
  inputs = [
      (
          temp_nodes[598],
          temp_nodes[494],
          jaxite_bool.constant(False, params),
          8,
      ),
      (temp_nodes[484], temp_nodes[180], temp_nodes[275], 128),
      (temp_nodes[708], temp_nodes[711], temp_nodes[726], 96),
  ]
  outputs = jaxite_bool.pmap_lut3(inputs, sks, params)
  temp_nodes[1044] = outputs[0]
  temp_nodes[1045] = outputs[1]
  temp_nodes[1546] = outputs[2]
  inputs = [
      (temp_nodes[580], temp_nodes[1593], temp_nodes[582], 11),
      (temp_nodes[1729], my_string[131], my_string[133], 128),
      (my_string[174], my_string[175], jaxite_bool.constant(False, params), 1),
      (temp_nodes[664], temp_nodes[624], temp_nodes[665], 64),
  ]
  outputs = jaxite_bool.pmap_lut3(inputs, sks, params)
  temp_nodes[1592] = outputs[0]
  temp_nodes[1728] = outputs[1]
  temp_nodes[63] = outputs[2]
  temp_nodes[623] = outputs[3]
  inputs = [
      (
          temp_nodes[670],
          temp_nodes[513],
          jaxite_bool.constant(False, params),
          1,
      ),
      (
          temp_nodes[670],
          temp_nodes[675],
          jaxite_bool.constant(False, params),
          1,
      ),
      (temp_nodes[670], temp_nodes[513], temp_nodes[684], 16),
      (temp_nodes[670], temp_nodes[1595], temp_nodes[495], 1),
  ]
  outputs = jaxite_bool.pmap_lut3(inputs, sks, params)
  temp_nodes[669] = outputs[0]
  temp_nodes[677] = outputs[1]
  temp_nodes[683] = outputs[2]
  temp_nodes[737] = outputs[3]
  inputs = [
      (temp_nodes[670], temp_nodes[1595], temp_nodes[739], 16),
      (my_string[129], my_string[128], my_string[130], 64),
      (temp_nodes[670], temp_nodes[513], temp_nodes[783], 16),
      (
          temp_nodes[625],
          temp_nodes[786],
          jaxite_bool.constant(False, params),
          4,
      ),
  ]
  outputs = jaxite_bool.pmap_lut3(inputs, sks, params)
  temp_nodes[738] = outputs[0]
  temp_nodes[755] = outputs[1]
  temp_nodes[782] = outputs[2]
  temp_nodes[785] = outputs[3]
  inputs = [
      (temp_nodes[580], temp_nodes[1593], temp_nodes[793], 176),
      (temp_nodes[798], my_string[173], jaxite_bool.constant(False, params), 8),
      (
          temp_nodes[825],
          temp_nodes[826],
          jaxite_bool.constant(False, params),
          8,
      ),
      (temp_nodes[1490], temp_nodes[663], temp_nodes[828], 64),
  ]
  outputs = jaxite_bool.pmap_lut3(inputs, sks, params)
  temp_nodes[792] = outputs[0]
  temp_nodes[797] = outputs[1]
  temp_nodes[824] = outputs[2]
  temp_nodes[827] = outputs[3]
  inputs = [
      (
          temp_nodes[629],
          temp_nodes[844],
          jaxite_bool.constant(False, params),
          4,
      ),
      (
          temp_nodes[432],
          temp_nodes[663],
          jaxite_bool.constant(False, params),
          8,
      ),
      (temp_nodes[828], my_string[112], jaxite_bool.constant(False, params), 8),
      (temp_nodes[175], temp_nodes[84], jaxite_bool.constant(False, params), 1),
  ]
  outputs = jaxite_bool.pmap_lut3(inputs, sks, params)
  temp_nodes[843] = outputs[0]
  temp_nodes[859] = outputs[1]
  temp_nodes[860] = outputs[2]
  temp_nodes[861] = outputs[3]
  inputs = [
      (
          temp_nodes[881],
          temp_nodes[885],
          jaxite_bool.constant(False, params),
          4,
      ),
      (temp_nodes[883], temp_nodes[822], temp_nodes[893], 96),
      (temp_nodes[896], temp_nodes[899], temp_nodes[473], 64),
      (
          temp_nodes[734],
          temp_nodes[910],
          jaxite_bool.constant(False, params),
          1,
      ),
  ]
  outputs = jaxite_bool.pmap_lut3(inputs, sks, params)
  temp_nodes[880] = outputs[0]
  temp_nodes[892] = outputs[1]
  temp_nodes[895] = outputs[2]
  temp_nodes[909] = outputs[3]
  inputs = [
      (temp_nodes[119], my_string[183], jaxite_bool.constant(False, params), 8),
      (
          temp_nodes[1707],
          my_string[119],
          jaxite_bool.constant(False, params),
          8,
      ),
      (
          temp_nodes[1602],
          temp_nodes[348],
          jaxite_bool.constant(False, params),
          4,
      ),
      (temp_nodes[442], temp_nodes[1044], temp_nodes[1045], 64),
  ]
  outputs = jaxite_bool.pmap_lut3(inputs, sks, params)
  temp_nodes[944] = outputs[0]
  temp_nodes[987] = outputs[1]
  temp_nodes[1041] = outputs[2]
  temp_nodes[1043] = outputs[3]
  inputs = [
      (temp_nodes[901], temp_nodes[1546], temp_nodes[558], 64),
      (temp_nodes[886], temp_nodes[817], temp_nodes[888], 112),
  ]
  outputs = jaxite_bool.pmap_lut3(inputs, sks, params)
  temp_nodes[1547] = outputs[0]
  temp_nodes[1548] = outputs[1]
  inputs = [
      (
          temp_nodes[442],
          temp_nodes[1592],
          jaxite_bool.constant(False, params),
          11,
      ),
      (temp_nodes[623], temp_nodes[669], temp_nodes[677], 128),
      (temp_nodes[626], my_string[117], jaxite_bool.constant(False, params), 8),
      (temp_nodes[442], temp_nodes[683], temp_nodes[1592], 64),
  ]
  outputs = jaxite_bool.pmap_lut3(inputs, sks, params)
  temp_nodes[1591] = outputs[0]
  temp_nodes[622] = outputs[1]
  temp_nodes[680] = outputs[2]
  temp_nodes[682] = outputs[3]
  inputs = [
      (temp_nodes[1710], temp_nodes[738], temp_nodes[737], 208),
      (temp_nodes[420], temp_nodes[1728], temp_nodes[755], 64),
      (temp_nodes[442], temp_nodes[782], temp_nodes[1592], 64),
      (temp_nodes[677], temp_nodes[785], temp_nodes[665], 128),
  ]
  outputs = jaxite_bool.pmap_lut3(inputs, sks, params)
  temp_nodes[736] = outputs[0]
  temp_nodes[754] = outputs[1]
  temp_nodes[781] = outputs[2]
  temp_nodes[784] = outputs[3]
  inputs = [
      (
          temp_nodes[792],
          temp_nodes[473],
          jaxite_bool.constant(False, params),
          8,
      ),
      (temp_nodes[797], temp_nodes[63], jaxite_bool.constant(False, params), 4),
      (temp_nodes[824], temp_nodes[827], temp_nodes[432], 128),
      (temp_nodes[442], temp_nodes[843], temp_nodes[1592], 64),
  ]
  outputs = jaxite_bool.pmap_lut3(inputs, sks, params)
  temp_nodes[791] = outputs[0]
  temp_nodes[796] = outputs[1]
  temp_nodes[823] = outputs[2]
  temp_nodes[842] = outputs[3]
  inputs = [
      (temp_nodes[1710], temp_nodes[738], temp_nodes[669], 208),
      (temp_nodes[1490], temp_nodes[859], temp_nodes[860], 64),
      (
          temp_nodes[880],
          temp_nodes[892],
          jaxite_bool.constant(False, params),
          8,
      ),
      (temp_nodes[1547], temp_nodes[895], my_string[180], 64),
  ]
  outputs = jaxite_bool.pmap_lut3(inputs, sks, params)
  temp_nodes[847] = outputs[0]
  temp_nodes[858] = outputs[1]
  temp_nodes[879] = outputs[2]
  temp_nodes[894] = outputs[3]
  inputs = [
      (temp_nodes[829], temp_nodes[861], temp_nodes[909], 128),
      (
          temp_nodes[1843],
          my_string[167],
          jaxite_bool.constant(False, params),
          4,
      ),
      (my_string[183], my_string[175], my_string[119], 1),
      (temp_nodes[154], my_string[191], jaxite_bool.constant(False, params), 4),
  ]
  outputs = jaxite_bool.pmap_lut3(inputs, sks, params)
  temp_nodes[908] = outputs[0]
  temp_nodes[912] = outputs[1]
  temp_nodes[913] = outputs[2]
  temp_nodes[942] = outputs[3]
  inputs = [
      (temp_nodes[154], my_string[191], jaxite_bool.constant(False, params), 8),
      (
          temp_nodes[987],
          temp_nodes[944],
          jaxite_bool.constant(False, params),
          1,
      ),
      (temp_nodes[1041], temp_nodes[741], temp_nodes[3], 208),
      (temp_nodes[495], temp_nodes[225], temp_nodes[1043], 16),
  ]
  outputs = jaxite_bool.pmap_lut3(inputs, sks, params)
  temp_nodes[945] = outputs[0]
  temp_nodes[986] = outputs[1]
  temp_nodes[1040] = outputs[2]
  temp_nodes[1042] = outputs[3]
  inputs = [
      (
          temp_nodes[434],
          temp_nodes[518],
          jaxite_bool.constant(False, params),
          4,
      ),
      (temp_nodes[887], temp_nodes[1548], temp_nodes[892], 64),
  ]
  outputs = jaxite_bool.pmap_lut3(inputs, sks, params)
  temp_nodes[1047] = outputs[0]
  temp_nodes[1549] = outputs[1]
  inputs = [
      (temp_nodes[622], temp_nodes[1591], temp_nodes[670], 13),
      (temp_nodes[622], temp_nodes[1591], temp_nodes[680], 208),
      (temp_nodes[682], temp_nodes[736], temp_nodes[754], 128),
      (temp_nodes[762], temp_nodes[143], temp_nodes[381], 16),
  ]
  outputs = jaxite_bool.pmap_lut3(inputs, sks, params)
  temp_nodes[1590] = outputs[0]
  temp_nodes[679] = outputs[1]
  temp_nodes[681] = outputs[2]
  temp_nodes[761] = outputs[3]
  inputs = [
      (my_string[136], my_string[141], jaxite_bool.constant(False, params), 8),
      (
          temp_nodes[781],
          temp_nodes[784],
          jaxite_bool.constant(False, params),
          8,
      ),
      (
          temp_nodes[580],
          temp_nodes[1593],
          jaxite_bool.constant(False, params),
          4,
      ),
      (temp_nodes[784], temp_nodes[781], temp_nodes[791], 112),
  ]
  outputs = jaxite_bool.pmap_lut3(inputs, sks, params)
  temp_nodes[765] = outputs[0]
  temp_nodes[780] = outputs[1]
  temp_nodes[788] = outputs[2]
  temp_nodes[790] = outputs[3]
  inputs = [
      (temp_nodes[796], temp_nodes[508], temp_nodes[823], 64),
      (temp_nodes[842], temp_nodes[847], temp_nodes[737], 128),
      (temp_nodes[796], temp_nodes[858], temp_nodes[824], 64),
      (
          temp_nodes[879],
          temp_nodes[894],
          jaxite_bool.constant(False, params),
          4,
      ),
  ]
  outputs = jaxite_bool.pmap_lut3(inputs, sks, params)
  temp_nodes[795] = outputs[0]
  temp_nodes[841] = outputs[1]
  temp_nodes[857] = outputs[2]
  temp_nodes[878] = outputs[3]
  inputs = [
      (
          temp_nodes[1496],
          temp_nodes[1504],
          jaxite_bool.constant(False, params),
          1,
      ),
      (temp_nodes[731], temp_nodes[912], temp_nodes[913], 16),
      (my_string[183], temp_nodes[942], jaxite_bool.constant(False, params), 1),
      (
          temp_nodes[944],
          temp_nodes[945],
          jaxite_bool.constant(False, params),
          1,
      ),
  ]
  outputs = jaxite_bool.pmap_lut3(inputs, sks, params)
  temp_nodes[906] = outputs[0]
  temp_nodes[911] = outputs[1]
  temp_nodes[941] = outputs[2]
  temp_nodes[943] = outputs[3]
  inputs = [
      (temp_nodes[1510], temp_nodes[986], temp_nodes[908], 64),
      (temp_nodes[812], temp_nodes[695], temp_nodes[701], 224),
      (
          temp_nodes[1040],
          temp_nodes[1042],
          jaxite_bool.constant(False, params),
          4,
      ),
      (temp_nodes[438], temp_nodes[531], temp_nodes[1047], 64),
  ]
  outputs = jaxite_bool.pmap_lut3(inputs, sks, params)
  temp_nodes[985] = outputs[0]
  temp_nodes[999] = outputs[1]
  temp_nodes[1039] = outputs[2]
  temp_nodes[1046] = outputs[3]
  inputs = [
      (temp_nodes[734], my_string[151], my_string[143], 1),
      (
          temp_nodes[826],
          temp_nodes[430],
          jaxite_bool.constant(False, params),
          8,
      ),
      (temp_nodes[575], temp_nodes[829], temp_nodes[660], 128),
      (temp_nodes[1922], temp_nodes[659], temp_nodes[910], 1),
  ]
  outputs = jaxite_bool.pmap_lut3(inputs, sks, params)
  temp_nodes[1067] = outputs[0]
  temp_nodes[1076] = outputs[1]
  temp_nodes[1077] = outputs[2]
  temp_nodes[1079] = outputs[3]
  inputs = [
      (
          temp_nodes[881],
          temp_nodes[1549],
          jaxite_bool.constant(False, params),
          4,
      ),
      (my_string[105], my_string[104], temp_nodes[620], 64),
  ]
  outputs = jaxite_bool.pmap_lut3(inputs, sks, params)
  temp_nodes[1550] = outputs[0]
  temp_nodes[1557] = outputs[1]
  inputs = [
      (temp_nodes[534], temp_nodes[1590], temp_nodes[1793], 112),
      (my_string[182], my_string[183], jaxite_bool.constant(False, params), 1),
      (temp_nodes[1705], temp_nodes[679], temp_nodes[681], 208),
      (
          temp_nodes[1591],
          temp_nodes[622],
          jaxite_bool.constant(False, params),
          4,
      ),
  ]
  outputs = jaxite_bool.pmap_lut3(inputs, sks, params)
  temp_nodes[1589] = outputs[0]
  temp_nodes[117] = outputs[1]
  temp_nodes[678] = outputs[2]
  temp_nodes[756] = outputs[3]
  inputs = [
      (
          temp_nodes[761],
          temp_nodes[422],
          jaxite_bool.constant(False, params),
          8,
      ),
      (my_string[137], temp_nodes[765], my_string[138], 64),
      (temp_nodes[780], temp_nodes[788], temp_nodes[515], 254),
      (temp_nodes[419], temp_nodes[790], temp_nodes[1729], 112),
  ]
  outputs = jaxite_bool.pmap_lut3(inputs, sks, params)
  temp_nodes[760] = outputs[0]
  temp_nodes[764] = outputs[1]
  temp_nodes[779] = outputs[2]
  temp_nodes[789] = outputs[3]
  inputs = [
      (temp_nodes[664], temp_nodes[1594], temp_nodes[795], 64),
      (temp_nodes[1705], temp_nodes[679], temp_nodes[841], 208),
      (temp_nodes[857], temp_nodes[1594], temp_nodes[508], 128),
      (temp_nodes[878], my_string[181], jaxite_bool.constant(False, params), 8),
  ]
  outputs = jaxite_bool.pmap_lut3(inputs, sks, params)
  temp_nodes[794] = outputs[0]
  temp_nodes[840] = outputs[1]
  temp_nodes[856] = outputs[2]
  temp_nodes[877] = outputs[3]
  inputs = [
      (temp_nodes[656], temp_nodes[908], temp_nodes[911], 128),
      (
          temp_nodes[1550],
          temp_nodes[879],
          jaxite_bool.constant(False, params),
          1,
      ),
      (temp_nodes[807], temp_nodes[802], temp_nodes[717], 1),
      (
          temp_nodes[941],
          temp_nodes[943],
          jaxite_bool.constant(False, params),
          8,
      ),
  ]
  outputs = jaxite_bool.pmap_lut3(inputs, sks, params)
  temp_nodes[907] = outputs[0]
  temp_nodes[935] = outputs[1]
  temp_nodes[936] = outputs[2]
  temp_nodes[940] = outputs[3]
  inputs = [
      (my_string[175], my_string[167], my_string[111], 1),
      (temp_nodes[824], temp_nodes[906], temp_nodes[985], 128),
      (
          temp_nodes[898],
          temp_nodes[999],
          jaxite_bool.constant(False, params),
          4,
      ),
      (temp_nodes[1597], temp_nodes[1046], temp_nodes[1039], 64),
  ]
  outputs = jaxite_bool.pmap_lut3(inputs, sks, params)
  temp_nodes[946] = outputs[0]
  temp_nodes[984] = outputs[1]
  temp_nodes[998] = outputs[2]
  temp_nodes[1038] = outputs[3]
  inputs = [
      (temp_nodes[1076], temp_nodes[825], temp_nodes[1077], 128),
      (
          temp_nodes[1490],
          temp_nodes[1079],
          jaxite_bool.constant(False, params),
          4,
      ),
      (temp_nodes[1557], temp_nodes[584], temp_nodes[610], 128),
      (my_string[127], my_string[119], temp_nodes[1067], 16),
  ]
  outputs = jaxite_bool.pmap_lut3(inputs, sks, params)
  temp_nodes[1075] = outputs[0]
  temp_nodes[1078] = outputs[1]
  temp_nodes[1558] = outputs[2]
  temp_nodes[1559] = outputs[3]
  inputs = [
      (temp_nodes[678], temp_nodes[1589], temp_nodes[756], 13),
      (
          temp_nodes[1595],
          temp_nodes[1590],
          jaxite_bool.constant(False, params),
          4,
      ),
      (temp_nodes[760], my_string[140], jaxite_bool.constant(False, params), 8),
      (temp_nodes[764], temp_nodes[1766], my_string[139], 128),
  ]
  outputs = jaxite_bool.pmap_lut3(inputs, sks, params)
  temp_nodes[1588] = outputs[0]
  temp_nodes[758] = outputs[1]
  temp_nodes[759] = outputs[2]
  temp_nodes[763] = outputs[3]
  inputs = [
      (temp_nodes[678], temp_nodes[1589], temp_nodes[1590], 208),
      (temp_nodes[678], temp_nodes[1589], temp_nodes[687], 208),
      (temp_nodes[678], temp_nodes[1589], temp_nodes[738], 208),
      (temp_nodes[779], temp_nodes[789], temp_nodes[794], 16),
  ]
  outputs = jaxite_bool.pmap_lut3(inputs, sks, params)
  temp_nodes[768] = outputs[0]
  temp_nodes[774] = outputs[1]
  temp_nodes[776] = outputs[2]
  temp_nodes[778] = outputs[3]
  inputs = [
      (temp_nodes[1589], temp_nodes[840], temp_nodes[754], 64),
      (
          temp_nodes[780],
          temp_nodes[788],
          jaxite_bool.constant(False, params),
          1,
      ),
      (
          temp_nodes[779],
          temp_nodes[789],
          jaxite_bool.constant(False, params),
          1,
      ),
      (temp_nodes[664], temp_nodes[737], temp_nodes[856], 64),
  ]
  outputs = jaxite_bool.pmap_lut3(inputs, sks, params)
  temp_nodes[839] = outputs[0]
  temp_nodes[848] = outputs[1]
  temp_nodes[854] = outputs[2]
  temp_nodes[855] = outputs[3]
  inputs = [
      (
          temp_nodes[877],
          temp_nodes[117],
          jaxite_bool.constant(False, params),
          4,
      ),
      (temp_nodes[1510], temp_nodes[906], temp_nodes[494], 64),
      (
          temp_nodes[935],
          temp_nodes[936],
          jaxite_bool.constant(False, params),
          8,
      ),
      (temp_nodes[940], temp_nodes[735], temp_nodes[946], 128),
  ]
  outputs = jaxite_bool.pmap_lut3(inputs, sks, params)
  temp_nodes[876] = outputs[0]
  temp_nodes[905] = outputs[1]
  temp_nodes[934] = outputs[2]
  temp_nodes[939] = outputs[3]
  inputs = [
      (
          temp_nodes[984],
          temp_nodes[907],
          jaxite_bool.constant(False, params),
          8,
      ),
      (
          temp_nodes[1038],
          temp_nodes[1558],
          jaxite_bool.constant(False, params),
          8,
      ),
      (temp_nodes[741], temp_nodes[587], temp_nodes[135], 64),
      (temp_nodes[519], temp_nodes[381], temp_nodes[419], 64),
  ]
  outputs = jaxite_bool.pmap_lut3(inputs, sks, params)
  temp_nodes[983] = outputs[0]
  temp_nodes[1037] = outputs[1]
  temp_nodes[1048] = outputs[2]
  temp_nodes[1050] = outputs[3]
  inputs = [
      (temp_nodes[1558], temp_nodes[1038], temp_nodes[739], 112),
      (temp_nodes[675], temp_nodes[1603], temp_nodes[568], 64),
      (temp_nodes[1075], temp_nodes[1078], temp_nodes[946], 128),
      (temp_nodes[884], temp_nodes[998], temp_nodes[897], 24),
  ]
  outputs = jaxite_bool.pmap_lut3(inputs, sks, params)
  temp_nodes[1055] = outputs[0]
  temp_nodes[1057] = outputs[1]
  temp_nodes[1074] = outputs[2]
  temp_nodes[1551] = outputs[3]
  inputs = [
      (temp_nodes[1559], temp_nodes[728], temp_nodes[729], 128),
  ]
  outputs = jaxite_bool.pmap_lut3(inputs, sks, params)
  temp_nodes[1560] = outputs[0]
  inputs = [
      (temp_nodes[630], temp_nodes[1588], temp_nodes[1811], 112),
      (
          temp_nodes[1793],
          my_string[147],
          jaxite_bool.constant(False, params),
          8,
      ),
      (temp_nodes[759], temp_nodes[758], temp_nodes[763], 112),
      (
          temp_nodes[762],
          temp_nodes[768],
          jaxite_bool.constant(False, params),
          4,
      ),
  ]
  outputs = jaxite_bool.pmap_lut3(inputs, sks, params)
  temp_nodes[1587] = outputs[0]
  temp_nodes[1792] = outputs[1]
  temp_nodes[757] = outputs[2]
  temp_nodes[767] = outputs[3]
  inputs = [
      (temp_nodes[567], temp_nodes[565], my_string[144], 128),
      (
          temp_nodes[774],
          temp_nodes[1841],
          jaxite_bool.constant(False, params),
          4,
      ),
      (temp_nodes[1710], temp_nodes[776], temp_nodes[1589], 13),
      (temp_nodes[778], temp_nodes[737], my_string[112], 128),
  ]
  outputs = jaxite_bool.pmap_lut3(inputs, sks, params)
  temp_nodes[771] = outputs[0]
  temp_nodes[773] = outputs[1]
  temp_nodes[775] = outputs[2]
  temp_nodes[777] = outputs[3]
  inputs = [
      (
          temp_nodes[715],
          temp_nodes[566],
          jaxite_bool.constant(False, params),
          8,
      ),
      (my_string[145], temp_nodes[210], temp_nodes[567], 64),
      (
          temp_nodes[839],
          temp_nodes[848],
          jaxite_bool.constant(False, params),
          4,
      ),
      (
          temp_nodes[776],
          temp_nodes[1710],
          jaxite_bool.constant(False, params),
          4,
      ),
  ]
  outputs = jaxite_bool.pmap_lut3(inputs, sks, params)
  temp_nodes[835] = outputs[0]
  temp_nodes[836] = outputs[1]
  temp_nodes[838] = outputs[2]
  temp_nodes[852] = outputs[3]
  inputs = [
      (temp_nodes[1589], temp_nodes[854], temp_nodes[855], 64),
      (temp_nodes[1841], temp_nodes[774], temp_nodes[1588], 208),
      (temp_nodes[1595], temp_nodes[768], temp_nodes[760], 64),
      (
          temp_nodes[1589],
          temp_nodes[678],
          jaxite_bool.constant(False, params),
          4,
      ),
  ]
  outputs = jaxite_bool.pmap_lut3(inputs, sks, params)
  temp_nodes[853] = outputs[0]
  temp_nodes[862] = outputs[1]
  temp_nodes[870] = outputs[2]
  temp_nodes[873] = outputs[3]
  inputs = [
      (
          temp_nodes[905],
          temp_nodes[907],
          jaxite_bool.constant(False, params),
          8,
      ),
      (temp_nodes[934], temp_nodes[806], my_string[188], 128),
      (temp_nodes[1494], temp_nodes[826], temp_nodes[939], 64),
      (
          temp_nodes[1510],
          temp_nodes[1516],
          jaxite_bool.constant(False, params),
          1,
      ),
  ]
  outputs = jaxite_bool.pmap_lut3(inputs, sks, params)
  temp_nodes[904] = outputs[0]
  temp_nodes[933] = outputs[1]
  temp_nodes[938] = outputs[2]
  temp_nodes[947] = outputs[3]
  inputs = [
      (temp_nodes[876], temp_nodes[983], temp_nodes[494], 64),
      (temp_nodes[1048], temp_nodes[1037], temp_nodes[1674], 208),
      (temp_nodes[1050], temp_nodes[1037], temp_nodes[1729], 208),
      (temp_nodes[1594], temp_nodes[508], temp_nodes[579], 128),
  ]
  outputs = jaxite_bool.pmap_lut3(inputs, sks, params)
  temp_nodes[982] = outputs[0]
  temp_nodes[1036] = outputs[1]
  temp_nodes[1049] = outputs[2]
  temp_nodes[1051] = outputs[3]
  inputs = [
      (temp_nodes[443], temp_nodes[1037], temp_nodes[1766], 208),
      (temp_nodes[1710], temp_nodes[1055], temp_nodes[495], 13),
      (temp_nodes[513], temp_nodes[515], temp_nodes[1057], 16),
      (temp_nodes[1074], temp_nodes[432], temp_nodes[177], 128),
  ]
  outputs = jaxite_bool.pmap_lut3(inputs, sks, params)
  temp_nodes[1053] = outputs[0]
  temp_nodes[1054] = outputs[1]
  temp_nodes[1056] = outputs[2]
  temp_nodes[1073] = outputs[3]
  inputs = [
      (temp_nodes[817], temp_nodes[880], temp_nodes[1551], 128),
      (temp_nodes[686], temp_nodes[132], temp_nodes[1560], 64),
  ]
  outputs = jaxite_bool.pmap_lut3(inputs, sks, params)
  temp_nodes[1552] = outputs[0]
  temp_nodes[1561] = outputs[1]
  inputs = [
      (
          temp_nodes[1587],
          temp_nodes[757],
          jaxite_bool.constant(False, params),
          4,
      ),
      (my_string[190], my_string[191], jaxite_bool.constant(False, params), 1),
      (temp_nodes[767], temp_nodes[769], temp_nodes[771], 128),
      (temp_nodes[773], temp_nodes[775], temp_nodes[777], 64),
  ]
  outputs = jaxite_bool.pmap_lut3(inputs, sks, params)
  temp_nodes[1586] = outputs[0]
  temp_nodes[152] = outputs[1]
  temp_nodes[766] = outputs[2]
  temp_nodes[772] = outputs[3]
  inputs = [
      (temp_nodes[767], temp_nodes[835], temp_nodes[836], 128),
      (
          temp_nodes[838],
          temp_nodes[535],
          jaxite_bool.constant(False, params),
          8,
      ),
      (temp_nodes[1792], my_string[146], my_string[149], 128),
      (temp_nodes[852], temp_nodes[853], temp_nodes[862], 64),
  ]
  outputs = jaxite_bool.pmap_lut3(inputs, sks, params)
  temp_nodes[834] = outputs[0]
  temp_nodes[837] = outputs[1]
  temp_nodes[849] = outputs[2]
  temp_nodes[851] = outputs[3]
  inputs = [
      (
          temp_nodes[870],
          temp_nodes[1765],
          jaxite_bool.constant(False, params),
          8,
      ),
      (temp_nodes[679], temp_nodes[873], temp_nodes[1705], 208),
      (temp_nodes[876], temp_nodes[904], temp_nodes[824], 64),
      (temp_nodes[933], my_string[189], jaxite_bool.constant(False, params), 8),
  ]
  outputs = jaxite_bool.pmap_lut3(inputs, sks, params)
  temp_nodes[869] = outputs[0]
  temp_nodes[872] = outputs[1]
  temp_nodes[875] = outputs[2]
  temp_nodes[932] = outputs[3]
  inputs = [
      (temp_nodes[728], temp_nodes[938], temp_nodes[947], 128),
      (temp_nodes[709], temp_nodes[446], temp_nodes[1547], 7),
      (temp_nodes[779], temp_nodes[982], temp_nodes[665], 64),
      (
          temp_nodes[1550],
          temp_nodes[1552],
          jaxite_bool.constant(False, params),
          1,
      ),
  ]
  outputs = jaxite_bool.pmap_lut3(inputs, sks, params)
  temp_nodes[937] = outputs[0]
  temp_nodes[956] = outputs[1]
  temp_nodes[981] = outputs[2]
  temp_nodes[997] = outputs[3]
  inputs = [
      (
          temp_nodes[896],
          temp_nodes[689],
          jaxite_bool.constant(False, params),
          1,
      ),
      (temp_nodes[264], my_string[199], jaxite_bool.constant(False, params), 8),
      (my_string[135], temp_nodes[280], jaxite_bool.constant(False, params), 4),
      (temp_nodes[264], my_string[199], jaxite_bool.constant(False, params), 4),
  ]
  outputs = jaxite_bool.pmap_lut3(inputs, sks, params)
  temp_nodes[1000] = outputs[0]
  temp_nodes[1008] = outputs[1]
  temp_nodes[1009] = outputs[2]
  temp_nodes[1014] = outputs[3]
  inputs = [
      (temp_nodes[1036], temp_nodes[1049], temp_nodes[1051], 16),
      (temp_nodes[1053], temp_nodes[1054], temp_nodes[1056], 64),
      (temp_nodes[1558], temp_nodes[1561], temp_nodes[79], 128),
      (temp_nodes[796], temp_nodes[1073], temp_nodes[663], 64),
  ]
  outputs = jaxite_bool.pmap_lut3(inputs, sks, params)
  temp_nodes[1035] = outputs[0]
  temp_nodes[1052] = outputs[1]
  temp_nodes[1066] = outputs[2]
  temp_nodes[1072] = outputs[3]
  inputs = [
      (temp_nodes[772], temp_nodes[1586], temp_nodes[766], 112),
      (
          temp_nodes[1811],
          my_string[155],
          jaxite_bool.constant(False, params),
          8,
      ),
      (temp_nodes[837], temp_nodes[834], temp_nodes[849], 64),
      (temp_nodes[851], temp_nodes[1586], temp_nodes[773], 7),
  ]
  outputs = jaxite_bool.pmap_lut3(inputs, sks, params)
  temp_nodes[1585] = outputs[0]
  temp_nodes[1810] = outputs[1]
  temp_nodes[833] = outputs[2]
  temp_nodes[850] = outputs[3]
  inputs = [
      (
          temp_nodes[1586],
          temp_nodes[772],
          jaxite_bool.constant(False, params),
          8,
      ),
      (
          temp_nodes[790],
          temp_nodes[419],
          jaxite_bool.constant(False, params),
          8,
      ),
      (
          temp_nodes[869],
          temp_nodes[1766],
          jaxite_bool.constant(False, params),
          4,
      ),
      (temp_nodes[875], temp_nodes[737], temp_nodes[665], 128),
  ]
  outputs = jaxite_bool.pmap_lut3(inputs, sks, params)
  temp_nodes[864] = outputs[0]
  temp_nodes[866] = outputs[1]
  temp_nodes[868] = outputs[2]
  temp_nodes[874] = outputs[3]
  inputs = [
      (temp_nodes[152], temp_nodes[932], temp_nodes[937], 208),
      (
          temp_nodes[839],
          temp_nodes[780],
          jaxite_bool.constant(False, params),
          1,
      ),
      (temp_nodes[900], temp_nodes[474], temp_nodes[956], 16),
      (
          temp_nodes[1587],
          temp_nodes[872],
          jaxite_bool.constant(False, params),
          1,
      ),
  ]
  outputs = jaxite_bool.pmap_lut3(inputs, sks, params)
  temp_nodes[931] = outputs[0]
  temp_nodes[953] = outputs[1]
  temp_nodes[955] = outputs[2]
  temp_nodes[979] = outputs[3]
  inputs = [
      (
          temp_nodes[981],
          temp_nodes[737],
          jaxite_bool.constant(False, params),
          8,
      ),
      (
          temp_nodes[997],
          temp_nodes[1000],
          jaxite_bool.constant(False, params),
          8,
      ),
      (temp_nodes[1008], temp_nodes[943], temp_nodes[1009], 64),
      (
          temp_nodes[659],
          temp_nodes[734],
          jaxite_bool.constant(False, params),
          1,
      ),
  ]
  outputs = jaxite_bool.pmap_lut3(inputs, sks, params)
  temp_nodes[980] = outputs[0]
  temp_nodes[996] = outputs[1]
  temp_nodes[1007] = outputs[2]
  temp_nodes[1010] = outputs[3]
  inputs = [
      (
          temp_nodes[661],
          temp_nodes[942],
          jaxite_bool.constant(False, params),
          1,
      ),
      (my_string[183], temp_nodes[119], temp_nodes[1014], 13),
      (temp_nodes[580], temp_nodes[1052], temp_nodes[1035], 64),
      (
          temp_nodes[515],
          temp_nodes[669],
          jaxite_bool.constant(False, params),
          4,
      ),
  ]
  outputs = jaxite_bool.pmap_lut3(inputs, sks, params)
  temp_nodes[1011] = outputs[0]
  temp_nodes[1013] = outputs[1]
  temp_nodes[1034] = outputs[2]
  temp_nodes[1058] = outputs[3]
  inputs = [
      (temp_nodes[1591], temp_nodes[664], temp_nodes[625], 1),
      (
          temp_nodes[679],
          temp_nodes[1705],
          jaxite_bool.constant(False, params),
          4,
      ),
      (
          temp_nodes[738],
          temp_nodes[1710],
          jaxite_bool.constant(False, params),
          4,
      ),
      (temp_nodes[664], temp_nodes[1066], temp_nodes[669], 64),
  ]
  outputs = jaxite_bool.pmap_lut3(inputs, sks, params)
  temp_nodes[1060] = outputs[0]
  temp_nodes[1063] = outputs[1]
  temp_nodes[1064] = outputs[2]
  temp_nodes[1065] = outputs[3]
  inputs = [
      (temp_nodes[1072], temp_nodes[737], temp_nodes[1051], 128),
      (my_string[207], temp_nodes[411], temp_nodes[1008], 7),
      (temp_nodes[411], my_string[207], jaxite_bool.constant(False, params), 4),
      (temp_nodes[630], temp_nodes[1811], temp_nodes[754], 176),
  ]
  outputs = jaxite_bool.pmap_lut3(inputs, sks, params)
  temp_nodes[1071] = outputs[0]
  temp_nodes[1137] = outputs[1]
  temp_nodes[1139] = outputs[2]
  temp_nodes[1553] = outputs[3]
  inputs = [
      (temp_nodes[1585], temp_nodes[833], temp_nodes[850], 128),
      (temp_nodes[797], temp_nodes[864], temp_nodes[63], 208),
      (temp_nodes[866], temp_nodes[864], temp_nodes[1729], 208),
      (temp_nodes[872], temp_nodes[779], temp_nodes[874], 16),
  ]
  outputs = jaxite_bool.pmap_lut3(inputs, sks, params)
  temp_nodes[1584] = outputs[0]
  temp_nodes[863] = outputs[1]
  temp_nodes[865] = outputs[2]
  temp_nodes[871] = outputs[3]
  inputs = [
      (
          temp_nodes[864],
          temp_nodes[839],
          jaxite_bool.constant(False, params),
          1,
      ),
      (
          temp_nodes[848],
          temp_nodes[534],
          jaxite_bool.constant(False, params),
          8,
      ),
      (temp_nodes[776], temp_nodes[864], temp_nodes[1710], 208),
      (temp_nodes[779], temp_nodes[931], temp_nodes[754], 64),
  ]
  outputs = jaxite_bool.pmap_lut3(inputs, sks, params)
  temp_nodes[926] = outputs[0]
  temp_nodes[927] = outputs[1]
  temp_nodes[929] = outputs[2]
  temp_nodes[930] = outputs[3]
  inputs = [
      (
          temp_nodes[864],
          temp_nodes[953],
          jaxite_bool.constant(False, params),
          4,
      ),
      (
          temp_nodes[476],
          temp_nodes[955],
          jaxite_bool.constant(False, params),
          4,
      ),
      (my_string[153], temp_nodes[239], jaxite_bool.constant(False, params), 4),
      (temp_nodes[1810], my_string[154], my_string[157], 128),
  ]
  outputs = jaxite_bool.pmap_lut3(inputs, sks, params)
  temp_nodes[952] = outputs[0]
  temp_nodes[954] = outputs[1]
  temp_nodes[957] = outputs[2]
  temp_nodes[960] = outputs[3]
  inputs = [
      (
          temp_nodes[1585],
          temp_nodes[833],
          jaxite_bool.constant(False, params),
          8,
      ),
      (temp_nodes[868], temp_nodes[979], temp_nodes[980], 64),
      (
          temp_nodes[996],
          temp_nodes[955],
          jaxite_bool.constant(False, params),
          8,
      ),
      (
          temp_nodes[1504],
          temp_nodes[1508],
          jaxite_bool.constant(False, params),
          1,
      ),
  ]
  outputs = jaxite_bool.pmap_lut3(inputs, sks, params)
  temp_nodes[977] = outputs[0]
  temp_nodes[978] = outputs[1]
  temp_nodes[995] = outputs[2]
  temp_nodes[1005] = outputs[3]
  inputs = [
      (temp_nodes[1007], temp_nodes[1010], temp_nodes[1011], 128),
      (temp_nodes[731], my_string[175], temp_nodes[1013], 16),
      (
          temp_nodes[1510],
          temp_nodes[1518],
          jaxite_bool.constant(False, params),
          1,
      ),
      (
          temp_nodes[1502],
          temp_nodes[1516],
          jaxite_bool.constant(False, params),
          1,
      ),
  ]
  outputs = jaxite_bool.pmap_lut3(inputs, sks, params)
  temp_nodes[1006] = outputs[0]
  temp_nodes[1012] = outputs[1]
  temp_nodes[1016] = outputs[2]
  temp_nodes[1017] = outputs[3]
  inputs = [
      (temp_nodes[1034], temp_nodes[1049], temp_nodes[1058], 16),
      (
          temp_nodes[1060],
          temp_nodes[665],
          jaxite_bool.constant(False, params),
          8,
      ),
      (temp_nodes[580], temp_nodes[677], temp_nodes[787], 64),
      (temp_nodes[1063], temp_nodes[1064], temp_nodes[1065], 16),
  ]
  outputs = jaxite_bool.pmap_lut3(inputs, sks, params)
  temp_nodes[1033] = outputs[0]
  temp_nodes[1059] = outputs[1]
  temp_nodes[1061] = outputs[2]
  temp_nodes[1062] = outputs[3]
  inputs = [
      (temp_nodes[773], temp_nodes[664], temp_nodes[775], 16),
      (temp_nodes[779], temp_nodes[789], temp_nodes[1071], 16),
      (temp_nodes[407], my_string[207], jaxite_bool.constant(False, params), 1),
      (temp_nodes[574], temp_nodes[734], temp_nodes[1137], 16),
  ]
  outputs = jaxite_bool.pmap_lut3(inputs, sks, params)
  temp_nodes[1069] = outputs[0]
  temp_nodes[1070] = outputs[1]
  temp_nodes[1134] = outputs[2]
  temp_nodes[1136] = outputs[3]
  inputs = [
      (temp_nodes[942], temp_nodes[1139], temp_nodes[1013], 16),
      (temp_nodes[1591], temp_nodes[1553], temp_nodes[737], 64),
  ]
  outputs = jaxite_bool.pmap_lut3(inputs, sks, params)
  temp_nodes[1138] = outputs[0]
  temp_nodes[1554] = outputs[1]
  inputs = [
      (temp_nodes[863], temp_nodes[865], temp_nodes[1584], 16),
      (temp_nodes[868], temp_nodes[1587], temp_nodes[871], 16),
      (
          temp_nodes[1586],
          temp_nodes[851],
          jaxite_bool.constant(False, params),
          8,
      ),
      (temp_nodes[927], temp_nodes[926], temp_nodes[1793], 112),
  ]
  outputs = jaxite_bool.pmap_lut3(inputs, sks, params)
  temp_nodes[1583] = outputs[0]
  temp_nodes[867] = outputs[1]
  temp_nodes[915] = outputs[2]
  temp_nodes[925] = outputs[3]
  inputs = [
      (temp_nodes[929], temp_nodes[872], temp_nodes[930], 16),
      (temp_nodes[952], temp_nodes[954], temp_nodes[957], 128),
      (temp_nodes[631], temp_nodes[952], temp_nodes[960], 112),
      (temp_nodes[952], temp_nodes[954], temp_nodes[239], 128),
  ]
  outputs = jaxite_bool.pmap_lut3(inputs, sks, params)
  temp_nodes[928] = outputs[0]
  temp_nodes[951] = outputs[1]
  temp_nodes[959] = outputs[2]
  temp_nodes[961] = outputs[3]
  inputs = [
      (temp_nodes[865], temp_nodes[977], temp_nodes[978], 64),
      (temp_nodes[863], temp_nodes[926], temp_nodes[850], 64),
      (temp_nodes[995], my_string[196], jaxite_bool.constant(False, params), 8),
      (temp_nodes[1005], temp_nodes[1006], temp_nodes[1012], 128),
  ]
  outputs = jaxite_bool.pmap_lut3(inputs, sks, params)
  temp_nodes[976] = outputs[0]
  temp_nodes[988] = outputs[1]
  temp_nodes[994] = outputs[2]
  temp_nodes[1004] = outputs[3]
  inputs = [
      (temp_nodes[1496], temp_nodes[1017], temp_nodes[1016], 64),
      (temp_nodes[66], temp_nodes[53], temp_nodes[417], 7),
      (temp_nodes[1033], temp_nodes[1059], temp_nodes[1061], 128),
      (temp_nodes[1586], temp_nodes[1069], temp_nodes[1070], 128),
  ]
  outputs = jaxite_bool.pmap_lut3(inputs, sks, params)
  temp_nodes[1015] = outputs[0]
  temp_nodes[1024] = outputs[1]
  temp_nodes[1032] = outputs[2]
  temp_nodes[1068] = outputs[3]
  inputs = [
      (temp_nodes[770], temp_nodes[762], temp_nodes[936], 16),
      (temp_nodes[1134], temp_nodes[943], temp_nodes[430], 64),
      (temp_nodes[572], temp_nodes[1138], temp_nodes[1136], 64),
      (temp_nodes[665], temp_nodes[904], temp_nodes[826], 128),
  ]
  outputs = jaxite_bool.pmap_lut3(inputs, sks, params)
  temp_nodes[1128] = outputs[0]
  temp_nodes[1133] = outputs[1]
  temp_nodes[1135] = outputs[2]
  temp_nodes[1159] = outputs[3]
  inputs = [
      (temp_nodes[640], temp_nodes[693], temp_nodes[643], 96),
      (temp_nodes[1589], temp_nodes[1554], temp_nodes[1062], 64),
  ]
  outputs = jaxite_bool.pmap_lut3(inputs, sks, params)
  temp_nodes[1162] = outputs[0]
  temp_nodes[1555] = outputs[1]
  inputs = [
      (
          temp_nodes[1583],
          temp_nodes[867],
          jaxite_bool.constant(False, params),
          8,
      ),
      (my_string[198], my_string[199], jaxite_bool.constant(False, params), 1),
      (
          temp_nodes[915],
          temp_nodes[873],
          jaxite_bool.constant(False, params),
          1,
      ),
      (temp_nodes[867], temp_nodes[1583], temp_nodes[869], 112),
  ]
  outputs = jaxite_bool.pmap_lut3(inputs, sks, params)
  temp_nodes[1582] = outputs[0]
  temp_nodes[262] = outputs[1]
  temp_nodes[914] = outputs[2]
  temp_nodes[920] = outputs[3]
  inputs = [
      (temp_nodes[867], temp_nodes[1583], temp_nodes[850], 112),
      (temp_nodes[867], temp_nodes[1583], temp_nodes[877], 112),
      (
          temp_nodes[925],
          temp_nodes[928],
          jaxite_bool.constant(False, params),
          4,
      ),
      (temp_nodes[867], temp_nodes[1583], temp_nodes[863], 7),
  ]
  outputs = jaxite_bool.pmap_lut3(inputs, sks, params)
  temp_nodes[921] = outputs[0]
  temp_nodes[923] = outputs[1]
  temp_nodes[924] = outputs[2]
  temp_nodes[949] = outputs[3]
  inputs = [
      (temp_nodes[867], temp_nodes[1583], temp_nodes[951], 112),
      (temp_nodes[959], temp_nodes[961], my_string[152], 128),
      (
          temp_nodes[976],
          temp_nodes[988],
          jaxite_bool.constant(False, params),
          8,
      ),
      (temp_nodes[994], my_string[197], jaxite_bool.constant(False, params), 8),
  ]
  outputs = jaxite_bool.pmap_lut3(inputs, sks, params)
  temp_nodes[950] = outputs[0]
  temp_nodes[958] = outputs[1]
  temp_nodes[975] = outputs[2]
  temp_nodes[993] = outputs[3]
  inputs = [
      (
          temp_nodes[1004],
          temp_nodes[1015],
          jaxite_bool.constant(False, params),
          8,
      ),
      (temp_nodes[762], temp_nodes[769], temp_nodes[1024], 64),
      (temp_nodes[961], my_string[152], jaxite_bool.constant(False, params), 8),
      (
          temp_nodes[951],
          temp_nodes[959],
          jaxite_bool.constant(False, params),
          8,
      ),
  ]
  outputs = jaxite_bool.pmap_lut3(inputs, sks, params)
  temp_nodes[1003] = outputs[0]
  temp_nodes[1023] = outputs[1]
  temp_nodes[1089] = outputs[2]
  temp_nodes[1090] = outputs[3]
  inputs = [
      (temp_nodes[935], temp_nodes[1128], my_string[204], 128),
      (
          temp_nodes[1500],
          temp_nodes[1016],
          jaxite_bool.constant(False, params),
          4,
      ),
      (temp_nodes[1516], temp_nodes[1133], temp_nodes[1135], 64),
      (temp_nodes[876], temp_nodes[1159], temp_nodes[825], 64),
  ]
  outputs = jaxite_bool.pmap_lut3(inputs, sks, params)
  temp_nodes[1127] = outputs[0]
  temp_nodes[1131] = outputs[1]
  temp_nodes[1132] = outputs[2]
  temp_nodes[1158] = outputs[3]
  inputs = [
      (temp_nodes[639], temp_nodes[692], temp_nodes[643], 96),
      (temp_nodes[1032], temp_nodes[1555], temp_nodes[1068], 11),
      (temp_nodes[1162], temp_nodes[640], temp_nodes[812], 197),
  ]
  outputs = jaxite_bool.pmap_lut3(inputs, sks, params)
  temp_nodes[1161] = outputs[0]
  temp_nodes[1556] = outputs[1]
  temp_nodes[1566] = outputs[2]
  inputs = [
      (
          temp_nodes[1841],
          my_string[163],
          jaxite_bool.constant(False, params),
          8,
      ),
      (my_string[206], my_string[207], jaxite_bool.constant(False, params), 1),
      (temp_nodes[1766], temp_nodes[920], temp_nodes[921], 208),
      (temp_nodes[117], temp_nodes[923], temp_nodes[924], 208),
  ]
  outputs = jaxite_bool.pmap_lut3(inputs, sks, params)
  temp_nodes[1840] = outputs[0]
  temp_nodes[409] = outputs[1]
  temp_nodes[919] = outputs[2]
  temp_nodes[922] = outputs[3]
  inputs = [
      (temp_nodes[949], temp_nodes[950], temp_nodes[958], 128),
      (
          temp_nodes[914],
          temp_nodes[1590],
          jaxite_bool.constant(False, params),
          8,
      ),
      (temp_nodes[975], temp_nodes[952], temp_nodes[630], 64),
      (temp_nodes[1582], temp_nodes[864], temp_nodes[866], 16),
  ]
  outputs = jaxite_bool.pmap_lut3(inputs, sks, params)
  temp_nodes[948] = outputs[0]
  temp_nodes[971] = outputs[1]
  temp_nodes[974] = outputs[2]
  temp_nodes[990] = outputs[3]
  inputs = [
      (
          temp_nodes[993],
          temp_nodes[262],
          jaxite_bool.constant(False, params),
          4,
      ),
      (
          temp_nodes[1003],
          temp_nodes[432],
          jaxite_bool.constant(False, params),
          8,
      ),
      (temp_nodes[717], temp_nodes[1023], temp_nodes[801], 64),
      (temp_nodes[1556], temp_nodes[688], temp_nodes[713], 128),
  ]
  outputs = jaxite_bool.pmap_lut3(inputs, sks, params)
  temp_nodes[992] = outputs[0]
  temp_nodes[1002] = outputs[1]
  temp_nodes[1022] = outputs[2]
  temp_nodes[1031] = outputs[3]
  inputs = [
      (
          temp_nodes[923],
          temp_nodes[117],
          jaxite_bool.constant(False, params),
          4,
      ),
      (temp_nodes[925], temp_nodes[929], temp_nodes[930], 16),
      (temp_nodes[1582], temp_nodes[1089], temp_nodes[1090], 64),
      (
          temp_nodes[920],
          temp_nodes[1766],
          jaxite_bool.constant(False, params),
          4,
      ),
  ]
  outputs = jaxite_bool.pmap_lut3(inputs, sks, params)
  temp_nodes[1085] = outputs[0]
  temp_nodes[1086] = outputs[1]
  temp_nodes[1088] = outputs[2]
  temp_nodes[1092] = outputs[3]
  inputs = [
      (
          temp_nodes[1127],
          my_string[205],
          jaxite_bool.constant(False, params),
          8,
      ),
      (temp_nodes[729], temp_nodes[1131], temp_nodes[1132], 128),
      (temp_nodes[868], temp_nodes[779], temp_nodes[1158], 16),
      (temp_nodes[1014], temp_nodes[1011], temp_nodes[832], 64),
  ]
  outputs = jaxite_bool.pmap_lut3(inputs, sks, params)
  temp_nodes[1126] = outputs[0]
  temp_nodes[1130] = outputs[1]
  temp_nodes[1157] = outputs[2]
  temp_nodes[1192] = outputs[3]
  inputs = [
      (temp_nodes[812], temp_nodes[1161], temp_nodes[1566], 96),
      (temp_nodes[810], temp_nodes[701], temp_nodes[814], 64),
  ]
  outputs = jaxite_bool.pmap_lut3(inputs, sks, params)
  temp_nodes[1567] = outputs[0]
  temp_nodes[1568] = outputs[1]
  inputs = [
      (temp_nodes[919], temp_nodes[922], temp_nodes[948], 128),
      (
          temp_nodes[971],
          temp_nodes[534],
          jaxite_bool.constant(False, params),
          8,
      ),
      (
          temp_nodes[974],
          temp_nodes[1811],
          jaxite_bool.constant(False, params),
          4,
      ),
      (
          temp_nodes[990],
          temp_nodes[1729],
          jaxite_bool.constant(False, params),
          4,
      ),
  ]
  outputs = jaxite_bool.pmap_lut3(inputs, sks, params)
  temp_nodes[918] = outputs[0]
  temp_nodes[970] = outputs[1]
  temp_nodes[973] = outputs[2]
  temp_nodes[989] = outputs[3]
  inputs = [
      (temp_nodes[625], temp_nodes[757], temp_nodes[1002], 64),
      (
          temp_nodes[914],
          temp_nodes[1022],
          jaxite_bool.constant(False, params),
          8,
      ),
      (my_string[161], temp_nodes[716], temp_nodes[632], 64),
      (
          temp_nodes[1582],
          temp_nodes[1031],
          jaxite_bool.constant(False, params),
          4,
      ),
  ]
  outputs = jaxite_bool.pmap_lut3(inputs, sks, params)
  temp_nodes[1001] = outputs[0]
  temp_nodes[1021] = outputs[1]
  temp_nodes[1028] = outputs[2]
  temp_nodes[1030] = outputs[3]
  inputs = [
      (
          temp_nodes[1840],
          my_string[165],
          jaxite_bool.constant(False, params),
          8,
      ),
      (
          temp_nodes[1085],
          temp_nodes[1086],
          jaxite_bool.constant(False, params),
          4,
      ),
      (temp_nodes[1088], temp_nodes[921], temp_nodes[949], 128),
      (
          temp_nodes[1092],
          temp_nodes[872],
          jaxite_bool.constant(False, params),
          1,
      ),
  ]
  outputs = jaxite_bool.pmap_lut3(inputs, sks, params)
  temp_nodes[1080] = outputs[0]
  temp_nodes[1084] = outputs[1]
  temp_nodes[1087] = outputs[2]
  temp_nodes[1091] = outputs[3]
  inputs = [
      (temp_nodes[992], temp_nodes[1003], temp_nodes[432], 64),
      (
          temp_nodes[1126],
          temp_nodes[409],
          jaxite_bool.constant(False, params),
          4,
      ),
      (
          temp_nodes[1130],
          temp_nodes[432],
          jaxite_bool.constant(False, params),
          8,
      ),
      (temp_nodes[992], temp_nodes[625], temp_nodes[1002], 16),
  ]
  outputs = jaxite_bool.pmap_lut3(inputs, sks, params)
  temp_nodes[1118] = outputs[0]
  temp_nodes[1125] = outputs[1]
  temp_nodes[1129] = outputs[2]
  temp_nodes[1149] = outputs[3]
  inputs = [
      (temp_nodes[865], temp_nodes[1556], temp_nodes[737], 64),
      (
          temp_nodes[872],
          temp_nodes[1157],
          jaxite_bool.constant(False, params),
          4,
      ),
      (temp_nodes[1518], temp_nodes[656], temp_nodes[1192], 64),
      (temp_nodes[945], temp_nodes[1139], my_string[215], 1),
  ]
  outputs = jaxite_bool.pmap_lut3(inputs, sks, params)
  temp_nodes[1155] = outputs[0]
  temp_nodes[1156] = outputs[1]
  temp_nodes[1191] = outputs[2]
  temp_nodes[1193] = outputs[3]
  inputs = [
      (
          temp_nodes[1533],
          my_string[215],
          jaxite_bool.constant(False, params),
          1,
      ),
      (temp_nodes[1587], temp_nodes[863], temp_nodes[1584], 16),
      (
          temp_nodes[1567],
          temp_nodes[1568],
          jaxite_bool.constant(False, params),
          8,
      ),
  ]
  outputs = jaxite_bool.pmap_lut3(inputs, sks, params)
  temp_nodes[1195] = outputs[0]
  temp_nodes[1562] = outputs[1]
  temp_nodes[1569] = outputs[2]
  inputs = [
      (
          temp_nodes[918],
          temp_nodes[932],
          jaxite_bool.constant(False, params),
          4,
      ),
      (temp_nodes[970], temp_nodes[918], temp_nodes[1793], 208),
      (temp_nodes[929], temp_nodes[992], temp_nodes[1001], 16),
      (temp_nodes[918], temp_nodes[1582], temp_nodes[1021], 16),
  ]
  outputs = jaxite_bool.pmap_lut3(inputs, sks, params)
  temp_nodes[968] = outputs[0]
  temp_nodes[969] = outputs[1]
  temp_nodes[991] = outputs[2]
  temp_nodes[1020] = outputs[3]
  inputs = [
      (
          temp_nodes[918],
          temp_nodes[1582],
          jaxite_bool.constant(False, params),
          1,
      ),
      (temp_nodes[802], temp_nodes[770], temp_nodes[1028], 16),
      (temp_nodes[1030], temp_nodes[1080], my_string[162], 64),
      (temp_nodes[1084], temp_nodes[1087], temp_nodes[1091], 128),
  ]
  outputs = jaxite_bool.pmap_lut3(inputs, sks, params)
  temp_nodes[1026] = outputs[0]
  temp_nodes[1027] = outputs[1]
  temp_nodes[1029] = outputs[2]
  temp_nodes[1083] = outputs[3]
  inputs = [
      (temp_nodes[973], temp_nodes[989], temp_nodes[625], 1),
      (
          temp_nodes[1118],
          temp_nodes[757],
          jaxite_bool.constant(False, params),
          8,
      ),
      (temp_nodes[929], temp_nodes[1149], temp_nodes[757], 64),
      (temp_nodes[1569], temp_nodes[688], temp_nodes[650], 64),
  ]
  outputs = jaxite_bool.pmap_lut3(inputs, sks, params)
  temp_nodes[1116] = outputs[0]
  temp_nodes[1117] = outputs[1]
  temp_nodes[1148] = outputs[2]
  temp_nodes[1160] = outputs[3]
  inputs = [
      (
          temp_nodes[770],
          temp_nodes[934],
          jaxite_bool.constant(False, params),
          4,
      ),
      (temp_nodes[1191], temp_nodes[1137], temp_nodes[1193], 128),
      (
          temp_nodes[1195],
          temp_nodes[1134],
          jaxite_bool.constant(False, params),
          1,
      ),
      (temp_nodes[992], temp_nodes[1125], temp_nodes[1129], 16),
  ]
  outputs = jaxite_bool.pmap_lut3(inputs, sks, params)
  temp_nodes[1187] = outputs[0]
  temp_nodes[1190] = outputs[1]
  temp_nodes[1194] = outputs[2]
  temp_nodes[1203] = outputs[3]
  inputs = [
      (temp_nodes[1562], temp_nodes[1155], temp_nodes[1156], 127),
  ]
  outputs = jaxite_bool.pmap_lut3(inputs, sks, params)
  temp_nodes[1563] = outputs[0]
  inputs = [
      (my_string[214], my_string[215], jaxite_bool.constant(False, params), 1),
      (
          temp_nodes[1582],
          temp_nodes[915],
          jaxite_bool.constant(False, params),
          1,
      ),
      (temp_nodes[152], temp_nodes[968], temp_nodes[969], 13),
      (temp_nodes[973], temp_nodes[989], temp_nodes[991], 16),
  ]
  outputs = jaxite_bool.pmap_lut3(inputs, sks, params)
  temp_nodes[460] = outputs[0]
  temp_nodes[963] = outputs[1]
  temp_nodes[967] = outputs[2]
  temp_nodes[972] = outputs[3]
  inputs = [
      (
          temp_nodes[1020],
          my_string[160],
          jaxite_bool.constant(False, params),
          8,
      ),
      (temp_nodes[1026], temp_nodes[914], temp_nodes[1027], 128),
      (temp_nodes[923], temp_nodes[1083], temp_nodes[117], 208),
      (
          temp_nodes[1083],
          temp_nodes[949],
          jaxite_bool.constant(False, params),
          4,
      ),
  ]
  outputs = jaxite_bool.pmap_lut3(inputs, sks, params)
  temp_nodes[1019] = outputs[0]
  temp_nodes[1025] = outputs[1]
  temp_nodes[1082] = outputs[2]
  temp_nodes[1093] = outputs[3]
  inputs = [
      (temp_nodes[1083], temp_nodes[975], temp_nodes[953], 16),
      (temp_nodes[929], temp_nodes[1116], temp_nodes[1117], 64),
      (temp_nodes[1020], temp_nodes[1029], my_string[160], 128),
      (
          temp_nodes[968],
          temp_nodes[152],
          jaxite_bool.constant(False, params),
          4,
      ),
  ]
  outputs = jaxite_bool.pmap_lut3(inputs, sks, params)
  temp_nodes[1094] = outputs[0]
  temp_nodes[1115] = outputs[1]
  temp_nodes[1119] = outputs[2]
  temp_nodes[1145] = outputs[3]
  inputs = [
      (temp_nodes[973], temp_nodes[989], temp_nodes[1148], 16),
      (
          temp_nodes[1187],
          temp_nodes[461],
          jaxite_bool.constant(False, params),
          8,
      ),
      (temp_nodes[1190], temp_nodes[1017], temp_nodes[1194], 128),
      (temp_nodes[1064], temp_nodes[1203], temp_nodes[977], 64),
  ]
  outputs = jaxite_bool.pmap_lut3(inputs, sks, params)
  temp_nodes[1147] = outputs[0]
  temp_nodes[1186] = outputs[1]
  temp_nodes[1189] = outputs[2]
  temp_nodes[1202] = outputs[3]
  inputs = [
      (temp_nodes[944], my_string[223], temp_nodes[571], 16),
      (temp_nodes[1068], temp_nodes[1563], temp_nodes[1160], 64),
  ]
  outputs = jaxite_bool.pmap_lut3(inputs, sks, params)
  temp_nodes[1230] = outputs[0]
  temp_nodes[1564] = outputs[1]
  inputs = [
      (temp_nodes[63], my_string[171], jaxite_bool.constant(False, params), 8),
      (
          temp_nodes[967],
          temp_nodes[972],
          jaxite_bool.constant(False, params),
          8,
      ),
      (temp_nodes[1019], temp_nodes[1025], temp_nodes[1029], 128),
      (temp_nodes[1082], temp_nodes[1093], temp_nodes[1094], 64),
  ]
  outputs = jaxite_bool.pmap_lut3(inputs, sks, params)
  temp_nodes[62] = outputs[0]
  temp_nodes[966] = outputs[1]
  temp_nodes[1018] = outputs[2]
  temp_nodes[1081] = outputs[3]
  inputs = [
      (temp_nodes[1082], temp_nodes[1093], temp_nodes[967], 64),
      (temp_nodes[1025], temp_nodes[1115], temp_nodes[1119], 128),
      (
          temp_nodes[1026],
          temp_nodes[850],
          jaxite_bool.constant(False, params),
          8,
      ),
      (temp_nodes[920], temp_nodes[918], temp_nodes[1766], 208),
  ]
  outputs = jaxite_bool.pmap_lut3(inputs, sks, params)
  temp_nodes[1113] = outputs[0]
  temp_nodes[1114] = outputs[1]
  temp_nodes[1121] = outputs[2]
  temp_nodes[1123] = outputs[3]
  inputs = [
      (temp_nodes[1125], temp_nodes[1064], temp_nodes[977], 16),
      (temp_nodes[1019], temp_nodes[1025], temp_nodes[1029], 128),
      (temp_nodes[1082], temp_nodes[1145], temp_nodes[1093], 16),
      (
          temp_nodes[969],
          temp_nodes[1147],
          jaxite_bool.constant(False, params),
          4,
      ),
  ]
  outputs = jaxite_bool.pmap_lut3(inputs, sks, params)
  temp_nodes[1124] = outputs[0]
  temp_nodes[1143] = outputs[1]
  temp_nodes[1144] = outputs[2]
  temp_nodes[1146] = outputs[3]
  inputs = [
      (
          temp_nodes[648],
          temp_nodes[252],
          jaxite_bool.constant(False, params),
          1,
      ),
      (temp_nodes[1083], temp_nodes[963], temp_nodes[798], 64),
      (my_string[170], my_string[168], my_string[173], 128),
      (
          temp_nodes[1186],
          temp_nodes[460],
          jaxite_bool.constant(False, params),
          4,
      ),
  ]
  outputs = jaxite_bool.pmap_lut3(inputs, sks, params)
  temp_nodes[1163] = outputs[0]
  temp_nodes[1165] = outputs[1]
  temp_nodes[1166] = outputs[2]
  temp_nodes[1185] = outputs[3]
  inputs = [
      (temp_nodes[1189], temp_nodes[494], temp_nodes[826], 128),
      (temp_nodes[1082], temp_nodes[989], temp_nodes[1202], 16),
      (temp_nodes[1139], my_string[215], temp_nodes[1230], 16),
      (
          temp_nodes[1083],
          temp_nodes[1564],
          jaxite_bool.constant(False, params),
          4,
      ),
  ]
  outputs = jaxite_bool.pmap_lut3(inputs, sks, params)
  temp_nodes[1188] = outputs[0]
  temp_nodes[1201] = outputs[1]
  temp_nodes[1229] = outputs[2]
  temp_nodes[1565] = outputs[3]
  inputs = [
      (temp_nodes[966], temp_nodes[1018], temp_nodes[1081], 128),
      (temp_nodes[1114], temp_nodes[1113], temp_nodes[974], 112),
      (temp_nodes[1123], temp_nodes[1124], temp_nodes[1129], 64),
      (temp_nodes[1143], temp_nodes[1144], temp_nodes[1146], 128),
  ]
  outputs = jaxite_bool.pmap_lut3(inputs, sks, params)
  temp_nodes[965] = outputs[0]
  temp_nodes[1112] = outputs[1]
  temp_nodes[1122] = outputs[2]
  temp_nodes[1142] = outputs[3]
  inputs = [
      (temp_nodes[1114], temp_nodes[1113], temp_nodes[633], 7),
      (my_string[169], temp_nodes[1163], temp_nodes[1565], 64),
      (temp_nodes[1165], temp_nodes[62], temp_nodes[1166], 64),
      (temp_nodes[1185], temp_nodes[1125], temp_nodes[1188], 16),
  ]
  outputs = jaxite_bool.pmap_lut3(inputs, sks, params)
  temp_nodes[1153] = outputs[0]
  temp_nodes[1154] = outputs[1]
  temp_nodes[1164] = outputs[2]
  temp_nodes[1184] = outputs[3]
  inputs = [
      (temp_nodes[1123], temp_nodes[1121], temp_nodes[1201], 64),
      (temp_nodes[1229], temp_nodes[1012], temp_nodes[1136], 128),
      (
          temp_nodes[1538],
          my_string[223],
          jaxite_bool.constant(False, params),
          1,
      ),
      (temp_nodes[561], my_string[223], jaxite_bool.constant(False, params), 4),
  ]
  outputs = jaxite_bool.pmap_lut3(inputs, sks, params)
  temp_nodes[1200] = outputs[0]
  temp_nodes[1228] = outputs[1]
  temp_nodes[1231] = outputs[2]
  temp_nodes[1286] = outputs[3]
  inputs = [
      (my_string[239], my_string[202], jaxite_bool.constant(False, params), 4),
  ]
  outputs = jaxite_bool.pmap_lut3(inputs, sks, params)
  temp_nodes[1341] = outputs[0]
  inputs = [
      (my_string[222], my_string[223], jaxite_bool.constant(False, params), 1),
      (
          temp_nodes[1112],
          temp_nodes[1811],
          jaxite_bool.constant(False, params),
          4,
      ),
      (temp_nodes[989], temp_nodes[1122], temp_nodes[1121], 64),
      (temp_nodes[968], temp_nodes[1142], temp_nodes[152], 208),
  ]
  outputs = jaxite_bool.pmap_lut3(inputs, sks, params)
  temp_nodes[559] = outputs[0]
  temp_nodes[1111] = outputs[1]
  temp_nodes[1120] = outputs[2]
  temp_nodes[1141] = outputs[3]
  inputs = [
      (
          temp_nodes[965],
          temp_nodes[1082],
          jaxite_bool.constant(False, params),
          1,
      ),
      (temp_nodes[965], temp_nodes[1083], temp_nodes[926], 16),
      (temp_nodes[1153], temp_nodes[1154], temp_nodes[1164], 128),
      (
          temp_nodes[1142],
          temp_nodes[1165],
          jaxite_bool.constant(False, params),
          4,
      ),
  ]
  outputs = jaxite_bool.pmap_lut3(inputs, sks, params)
  temp_nodes[1150] = outputs[0]
  temp_nodes[1151] = outputs[1]
  temp_nodes[1152] = outputs[2]
  temp_nodes[1176] = outputs[3]
  inputs = [
      (
          temp_nodes[1113],
          temp_nodes[1114],
          jaxite_bool.constant(False, params),
          8,
      ),
      (
          temp_nodes[1083],
          temp_nodes[970],
          jaxite_bool.constant(False, params),
          4,
      ),
      (
          temp_nodes[950],
          temp_nodes[958],
          jaxite_bool.constant(False, params),
          8,
      ),
      (temp_nodes[789], temp_nodes[1184], temp_nodes[850], 64),
  ]
  outputs = jaxite_bool.pmap_lut3(inputs, sks, params)
  temp_nodes[1179] = outputs[0]
  temp_nodes[1180] = outputs[1]
  temp_nodes[1182] = outputs[2]
  temp_nodes[1183] = outputs[3]
  inputs = [
      (temp_nodes[1811], temp_nodes[1112], temp_nodes[1200], 208),
      (
          temp_nodes[934],
          temp_nodes[560],
          jaxite_bool.constant(False, params),
          8,
      ),
      (temp_nodes[1231], temp_nodes[1228], temp_nodes[1005], 64),
      (
          temp_nodes[1502],
          temp_nodes[1194],
          jaxite_bool.constant(False, params),
          4,
      ),
  ]
  outputs = jaxite_bool.pmap_lut3(inputs, sks, params)
  temp_nodes[1199] = outputs[0]
  temp_nodes[1225] = outputs[1]
  temp_nodes[1227] = outputs[2]
  temp_nodes[1232] = outputs[3]
  inputs = [
      (
          temp_nodes[807],
          temp_nodes[935],
          jaxite_bool.constant(False, params),
          4,
      ),
      (my_string[231], temp_nodes[647], temp_nodes[1286], 13),
      (temp_nodes[1341], my_string[203], my_string[205], 128),
  ]
  outputs = jaxite_bool.pmap_lut3(inputs, sks, params)
  temp_nodes[1243] = outputs[0]
  temp_nodes[1285] = outputs[1]
  temp_nodes[1340] = outputs[2]
  inputs = [
      (temp_nodes[117], my_string[179], jaxite_bool.constant(False, params), 8),
      (temp_nodes[1111], temp_nodes[992], temp_nodes[1120], 16),
      (temp_nodes[1141], temp_nodes[1150], temp_nodes[1151], 64),
      (
          temp_nodes[1176],
          my_string[173],
          jaxite_bool.constant(False, params),
          8,
      ),
  ]
  outputs = jaxite_bool.pmap_lut3(inputs, sks, params)
  temp_nodes[116] = outputs[0]
  temp_nodes[1110] = outputs[1]
  temp_nodes[1140] = outputs[2]
  temp_nodes[1175] = outputs[3]
  inputs = [
      (temp_nodes[1180], temp_nodes[1179], temp_nodes[1793], 208),
      (temp_nodes[1123], temp_nodes[1182], temp_nodes[1183], 64),
      (temp_nodes[1141], temp_nodes[1152], temp_nodes[1199], 64),
      (my_string[177], temp_nodes[895], jaxite_bool.constant(False, params), 4),
  ]
  outputs = jaxite_bool.pmap_lut3(inputs, sks, params)
  temp_nodes[1178] = outputs[0]
  temp_nodes[1181] = outputs[1]
  temp_nodes[1198] = outputs[2]
  temp_nodes[1207] = outputs[3]
  inputs = [
      (my_string[178], my_string[176], my_string[181], 128),
      (
          temp_nodes[1225],
          temp_nodes[559],
          jaxite_bool.constant(False, params),
          4,
      ),
      (temp_nodes[1131], temp_nodes[1227], temp_nodes[1232], 128),
      (
          temp_nodes[802],
          temp_nodes[1243],
          jaxite_bool.constant(False, params),
          4,
      ),
  ]
  outputs = jaxite_bool.pmap_lut3(inputs, sks, params)
  temp_nodes[1209] = outputs[0]
  temp_nodes[1224] = outputs[1]
  temp_nodes[1226] = outputs[2]
  temp_nodes[1242] = outputs[3]
  inputs = [
      (temp_nodes[717], temp_nodes[762], temp_nodes[769], 16),
      (temp_nodes[1185], temp_nodes[789], temp_nodes[1188], 16),
      (
          temp_nodes[1083],
          temp_nodes[975],
          jaxite_bool.constant(False, params),
          1,
      ),
      (temp_nodes[895], my_string[176], jaxite_bool.constant(False, params), 8),
  ]
  outputs = jaxite_bool.pmap_lut3(inputs, sks, params)
  temp_nodes[1244] = outputs[0]
  temp_nodes[1259] = outputs[1]
  temp_nodes[1261] = outputs[2]
  temp_nodes[1264] = outputs[3]
  inputs = [
      (
          my_string[215],
          temp_nodes[1285],
          jaxite_bool.constant(False, params),
          4,
      ),
      (temp_nodes[561], my_string[223], jaxite_bool.constant(False, params), 8),
      (
          temp_nodes[1340],
          temp_nodes[409],
          jaxite_bool.constant(False, params),
          8,
      ),
  ]
  outputs = jaxite_bool.pmap_lut3(inputs, sks, params)
  temp_nodes[1284] = outputs[0]
  temp_nodes[1289] = outputs[1]
  temp_nodes[1339] = outputs[2]
  inputs = [
      (temp_nodes[1110], temp_nodes[1140], temp_nodes[1152], 128),
      (
          temp_nodes[1175],
          temp_nodes[63],
          jaxite_bool.constant(False, params),
          4,
      ),
      (temp_nodes[1141], temp_nodes[1178], temp_nodes[1181], 16),
      (temp_nodes[1198], temp_nodes[1142], temp_nodes[993], 16),
  ]
  outputs = jaxite_bool.pmap_lut3(inputs, sks, params)
  temp_nodes[1109] = outputs[0]
  temp_nodes[1174] = outputs[1]
  temp_nodes[1177] = outputs[2]
  temp_nodes[1197] = outputs[3]
  inputs = [
      (temp_nodes[1198], temp_nodes[1142], temp_nodes[963], 16),
      (temp_nodes[1142], temp_nodes[1026], temp_nodes[878], 64),
      (
          temp_nodes[116],
          temp_nodes[1209],
          jaxite_bool.constant(False, params),
          8,
      ),
      (temp_nodes[868], temp_nodes[1224], temp_nodes[1226], 16),
  ]
  outputs = jaxite_bool.pmap_lut3(inputs, sks, params)
  temp_nodes[1204] = outputs[0]
  temp_nodes[1206] = outputs[1]
  temp_nodes[1208] = outputs[2]
  temp_nodes[1223] = outputs[3]
  inputs = [
      (
          temp_nodes[1242],
          temp_nodes[1244],
          jaxite_bool.constant(False, params),
          8,
      ),
      (temp_nodes[1123], temp_nodes[1088], temp_nodes[1259], 64),
      (temp_nodes[965], temp_nodes[1552], temp_nodes[1547], 1),
      (temp_nodes[1264], temp_nodes[116], my_string[178], 128),
  ]
  outputs = jaxite_bool.pmap_lut3(inputs, sks, params)
  temp_nodes[1241] = outputs[0]
  temp_nodes[1258] = outputs[1]
  temp_nodes[1260] = outputs[2]
  temp_nodes[1263] = outputs[3]
  inputs = [
      (
          temp_nodes[1139],
          temp_nodes[1284],
          jaxite_bool.constant(False, params),
          4,
      ),
      (my_string[231], temp_nodes[647], temp_nodes[1289], 7),
      (temp_nodes[731], my_string[175], temp_nodes[909], 16),
      (temp_nodes[912], my_string[191], temp_nodes[1339], 16),
  ]
  outputs = jaxite_bool.pmap_lut3(inputs, sks, params)
  temp_nodes[1283] = outputs[0]
  temp_nodes[1288] = outputs[1]
  temp_nodes[1336] = outputs[2]
  temp_nodes[1338] = outputs[3]
  inputs = [
      (temp_nodes[850], temp_nodes[1207], temp_nodes[1261], 128),
  ]
  outputs = jaxite_bool.pmap_lut3(inputs, sks, params)
  temp_nodes[1573] = outputs[0]
  inputs = [
      (temp_nodes[152], my_string[187], jaxite_bool.constant(False, params), 8),
      (
          temp_nodes[1174],
          temp_nodes[1177],
          jaxite_bool.constant(False, params),
          4,
      ),
      (temp_nodes[262], temp_nodes[1197], temp_nodes[1204], 208),
      (temp_nodes[1206], temp_nodes[1207], temp_nodes[1208], 64),
  ]
  outputs = jaxite_bool.pmap_lut3(inputs, sks, params)
  temp_nodes[151] = outputs[0]
  temp_nodes[1173] = outputs[1]
  temp_nodes[1196] = outputs[2]
  temp_nodes[1205] = outputs[3]
  inputs = [
      (
          temp_nodes[1109],
          temp_nodes[965],
          jaxite_bool.constant(False, params),
          1,
      ),
      (
          temp_nodes[1197],
          temp_nodes[262],
          jaxite_bool.constant(False, params),
          4,
      ),
      (temp_nodes[1112], temp_nodes[1198], temp_nodes[1811], 208),
      (temp_nodes[1178], temp_nodes[1143], temp_nodes[1223], 64),
  ]
  outputs = jaxite_bool.pmap_lut3(inputs, sks, params)
  temp_nodes[1218] = outputs[0]
  temp_nodes[1219] = outputs[1]
  temp_nodes[1221] = outputs[2]
  temp_nodes[1222] = outputs[3]
  inputs = [
      (
          temp_nodes[1142],
          temp_nodes[918],
          jaxite_bool.constant(False, params),
          1,
      ),
      (
          temp_nodes[417],
          temp_nodes[1241],
          jaxite_bool.constant(False, params),
          4,
      ),
      (
          temp_nodes[965],
          temp_nodes[1083],
          jaxite_bool.constant(False, params),
          1,
      ),
      (temp_nodes[1174], temp_nodes[1178], temp_nodes[1258], 16),
  ]
  outputs = jaxite_bool.pmap_lut3(inputs, sks, params)
  temp_nodes[1239] = outputs[0]
  temp_nodes[1240] = outputs[1]
  temp_nodes[1247] = outputs[2]
  temp_nodes[1257] = outputs[3]
  inputs = [
      (temp_nodes[1206], temp_nodes[1263], my_string[181], 64),
      (temp_nodes[1231], temp_nodes[644], temp_nodes[1232], 16),
      (temp_nodes[1283], temp_nodes[940], temp_nodes[1010], 128),
      (my_string[207], temp_nodes[411], temp_nodes[1288], 112),
  ]
  outputs = jaxite_bool.pmap_lut3(inputs, sks, params)
  temp_nodes[1262] = outputs[0]
  temp_nodes[1280] = outputs[1]
  temp_nodes[1282] = outputs[2]
  temp_nodes[1287] = outputs[3]
  inputs = [
      (
          temp_nodes[690],
          temp_nodes[1161],
          jaxite_bool.constant(False, params),
          8,
      ),
      (temp_nodes[1504], temp_nodes[1288], temp_nodes[1336], 64),
      (
          my_string[199],
          temp_nodes[1338],
          jaxite_bool.constant(False, params),
          4,
      ),
      (
          temp_nodes[644],
          temp_nodes[702],
          jaxite_bool.constant(False, params),
          1,
      ),
  ]
  outputs = jaxite_bool.pmap_lut3(inputs, sks, params)
  temp_nodes[1329] = outputs[0]
  temp_nodes[1335] = outputs[1]
  temp_nodes[1337] = outputs[2]
  temp_nodes[1343] = outputs[3]
  inputs = [
      (temp_nodes[1141], temp_nodes[1573], temp_nodes[1260], 64),
  ]
  outputs = jaxite_bool.pmap_lut3(inputs, sks, params)
  temp_nodes[1574] = outputs[0]
  inputs = [
      (temp_nodes[1173], temp_nodes[1196], temp_nodes[1205], 128),
      (
          temp_nodes[1082],
          temp_nodes[1218],
          jaxite_bool.constant(False, params),
          11,
      ),
      (temp_nodes[1174], temp_nodes[1221], temp_nodes[1222], 16),
      (
          temp_nodes[1198],
          temp_nodes[1026],
          jaxite_bool.constant(False, params),
          4,
      ),
  ]
  outputs = jaxite_bool.pmap_lut3(inputs, sks, params)
  temp_nodes[1172] = outputs[0]
  temp_nodes[1217] = outputs[1]
  temp_nodes[1220] = outputs[2]
  temp_nodes[1235] = outputs[3]
  inputs = [
      (my_string[185], temp_nodes[1240], temp_nodes[1239], 64),
      (
          temp_nodes[1247],
          temp_nodes[933],
          jaxite_bool.constant(False, params),
          8,
      ),
      (temp_nodes[151], my_string[186], my_string[189], 128),
      (temp_nodes[1219], temp_nodes[1125], temp_nodes[1257], 16),
  ]
  outputs = jaxite_bool.pmap_lut3(inputs, sks, params)
  temp_nodes[1238] = outputs[0]
  temp_nodes[1246] = outputs[1]
  temp_nodes[1248] = outputs[2]
  temp_nodes[1256] = outputs[3]
  inputs = [
      (temp_nodes[868], temp_nodes[1224], temp_nodes[1018], 16),
      (temp_nodes[996], my_string[228], my_string[229], 128),
      (my_string[230], my_string[231], jaxite_bool.constant(False, params), 1),
      (
          temp_nodes[1280],
          temp_nodes[729],
          jaxite_bool.constant(False, params),
          8,
      ),
  ]
  outputs = jaxite_bool.pmap_lut3(inputs, sks, params)
  temp_nodes[1270] = outputs[0]
  temp_nodes[1277] = outputs[1]
  temp_nodes[1278] = outputs[2]
  temp_nodes[1279] = outputs[3]
  inputs = [
      (temp_nodes[661], temp_nodes[1287], temp_nodes[1282], 64),
      (temp_nodes[1569], temp_nodes[1329], temp_nodes[633], 1),
      (temp_nodes[1243], my_string[236], my_string[237], 128),
      (temp_nodes[1335], temp_nodes[1284], temp_nodes[1337], 128),
  ]
  outputs = jaxite_bool.pmap_lut3(inputs, sks, params)
  temp_nodes[1281] = outputs[0]
  temp_nodes[1328] = outputs[1]
  temp_nodes[1332] = outputs[2]
  temp_nodes[1334] = outputs[3]
  inputs = [
      (temp_nodes[1231], temp_nodes[826], temp_nodes[1343], 64),
      (temp_nodes[1516], temp_nodes[1518], temp_nodes[1195], 1),
      (my_string[209], my_string[210], temp_nodes[460], 64),
      (my_string[231], my_string[218], jaxite_bool.constant(False, params), 4),
  ]
  outputs = jaxite_bool.pmap_lut3(inputs, sks, params)
  temp_nodes[1342] = outputs[0]
  temp_nodes[1344] = outputs[1]
  temp_nodes[1383] = outputs[2]
  temp_nodes[1416] = outputs[3]
  inputs = [
      (temp_nodes[1109], temp_nodes[1574], temp_nodes[1262], 64),
  ]
  outputs = jaxite_bool.pmap_lut3(inputs, sks, params)
  temp_nodes[1575] = outputs[0]
  inputs = [
      (temp_nodes[1217], temp_nodes[1219], temp_nodes[1220], 16),
      (temp_nodes[1186], temp_nodes[1172], temp_nodes[460], 208),
      (
          temp_nodes[1172],
          temp_nodes[1235],
          jaxite_bool.constant(False, params),
          4,
      ),
      (temp_nodes[1172], temp_nodes[1198], temp_nodes[1238], 16),
  ]
  outputs = jaxite_bool.pmap_lut3(inputs, sks, params)
  temp_nodes[1216] = outputs[0]
  temp_nodes[1233] = outputs[1]
  temp_nodes[1234] = outputs[2]
  temp_nodes[1237] = outputs[3]
  inputs = [
      (temp_nodes[1246], temp_nodes[1198], temp_nodes[1248], 208),
      (
          temp_nodes[1172],
          temp_nodes[1198],
          jaxite_bool.constant(False, params),
          1,
      ),
      (
          temp_nodes[1109],
          temp_nodes[1261],
          jaxite_bool.constant(False, params),
          4,
      ),
      (temp_nodes[1178], temp_nodes[1270], temp_nodes[1226], 64),
  ]
  outputs = jaxite_bool.pmap_lut3(inputs, sks, params)
  temp_nodes[1245] = outputs[0]
  temp_nodes[1250] = outputs[1]
  temp_nodes[1265] = outputs[2]
  temp_nodes[1269] = outputs[3]
  inputs = [
      (
          temp_nodes[1277],
          temp_nodes[1278],
          jaxite_bool.constant(False, params),
          4,
      ),
      (temp_nodes[947], temp_nodes[1279], temp_nodes[1281], 128),
      (
          temp_nodes[1328],
          temp_nodes[650],
          jaxite_bool.constant(False, params),
          8,
      ),
      (temp_nodes[1332], my_string[238], my_string[239], 1),
  ]
  outputs = jaxite_bool.pmap_lut3(inputs, sks, params)
  temp_nodes[1276] = outputs[0]
  temp_nodes[1318] = outputs[1]
  temp_nodes[1327] = outputs[2]
  temp_nodes[1331] = outputs[3]
  inputs = [
      (temp_nodes[1334], temp_nodes[1342], temp_nodes[1344], 128),
      (my_string[247], my_string[239], temp_nodes[1383], 16),
      (my_string[212], my_string[211], my_string[213], 64),
      (temp_nodes[65], my_string[175], jaxite_bool.constant(False, params), 4),
  ]
  outputs = jaxite_bool.pmap_lut3(inputs, sks, params)
  temp_nodes[1333] = outputs[0]
  temp_nodes[1382] = outputs[1]
  temp_nodes[1384] = outputs[2]
  temp_nodes[1386] = outputs[3]
  inputs = [
      (my_string[247], my_string[239], temp_nodes[559], 16),
      (my_string[254], my_string[255], temp_nodes[1416], 16),
      (
          temp_nodes[1575],
          temp_nodes[1256],
          jaxite_bool.constant(False, params),
          8,
      ),
  ]
  outputs = jaxite_bool.pmap_lut3(inputs, sks, params)
  temp_nodes[1414] = outputs[0]
  temp_nodes[1415] = outputs[1]
  temp_nodes[1576] = outputs[2]
  inputs = [
      (
          temp_nodes[1482],
          temp_nodes[1936],
          jaxite_bool.constant(False, params),
          4,
      ),
      (temp_nodes[180], temp_nodes[170], temp_nodes[172], 128),
      (temp_nodes[1233], temp_nodes[1216], temp_nodes[1234], 64),
      (temp_nodes[1237], temp_nodes[1245], my_string[184], 128),
  ]
  outputs = jaxite_bool.pmap_lut3(inputs, sks, params)
  temp_nodes[1106] = outputs[0]
  temp_nodes[1107] = outputs[1]
  temp_nodes[1215] = outputs[2]
  temp_nodes[1236] = outputs[3]
  inputs = [
      (temp_nodes[1126], temp_nodes[1250], temp_nodes[409], 112),
      (temp_nodes[1175], temp_nodes[1576], temp_nodes[63], 208),
      (temp_nodes[1217], temp_nodes[1221], temp_nodes[1269], 16),
      (temp_nodes[925], temp_nodes[1276], temp_nodes[1318], 16),
  ]
  outputs = jaxite_bool.pmap_lut3(inputs, sks, params)
  temp_nodes[1249] = outputs[0]
  temp_nodes[1267] = outputs[1]
  temp_nodes[1268] = outputs[2]
  temp_nodes[1317] = outputs[3]
  inputs = [
      (my_string[201], temp_nodes[1327], temp_nodes[1250], 64),
      (temp_nodes[1276], temp_nodes[1331], temp_nodes[1333], 16),
      (
          temp_nodes[1382],
          temp_nodes[1384],
          jaxite_bool.constant(False, params),
          8,
      ),
      (temp_nodes[1008], temp_nodes[1139], temp_nodes[1386], 1),
  ]
  outputs = jaxite_bool.pmap_lut3(inputs, sks, params)
  temp_nodes[1326] = outputs[0]
  temp_nodes[1330] = outputs[1]
  temp_nodes[1381] = outputs[2]
  temp_nodes[1385] = outputs[3]
  inputs = [
      (
          temp_nodes[1414],
          temp_nodes[1415],
          jaxite_bool.constant(False, params),
          8,
      ),
      (my_string[207], temp_nodes[411], temp_nodes[1386], 7),
      (my_string[217], my_string[219], my_string[221], 64),
      (temp_nodes[1233], temp_nodes[1576], temp_nodes[1265], 16),
  ]
  outputs = jaxite_bool.pmap_lut3(inputs, sks, params)
  temp_nodes[1413] = outputs[0]
  temp_nodes[1417] = outputs[1]
  temp_nodes[1418] = outputs[2]
  temp_nodes[1570] = outputs[3]
  inputs = [
      (temp_nodes[262], my_string[195], jaxite_bool.constant(False, params), 8),
      (temp_nodes[1484], temp_nodes[1107], temp_nodes[1106], 64),
      (temp_nodes[1249], temp_nodes[1236], temp_nodes[1215], 64),
      (temp_nodes[1249], temp_nodes[1267], temp_nodes[1268], 16),
  ]
  outputs = jaxite_bool.pmap_lut3(inputs, sks, params)
  temp_nodes[261] = outputs[0]
  temp_nodes[1105] = outputs[1]
  temp_nodes[1214] = outputs[2]
  temp_nodes[1266] = outputs[3]
  inputs = [
      (
          temp_nodes[1141],
          temp_nodes[1250],
          jaxite_bool.constant(False, params),
          4,
      ),
      (
          temp_nodes[1172],
          temp_nodes[1186],
          jaxite_bool.constant(False, params),
          4,
      ),
      (
          temp_nodes[1198],
          temp_nodes[1142],
          jaxite_bool.constant(False, params),
          1,
      ),
      (temp_nodes[1082], temp_nodes[1152], temp_nodes[1317], 64),
  ]
  outputs = jaxite_bool.pmap_lut3(inputs, sks, params)
  temp_nodes[1273] = outputs[0]
  temp_nodes[1293] = outputs[1]
  temp_nodes[1298] = outputs[2]
  temp_nodes[1316] = outputs[3]
  inputs = [
      (temp_nodes[1326], temp_nodes[1205], temp_nodes[1330], 128),
      (temp_nodes[1250], temp_nodes[1327], my_string[200], 128),
      (my_string[167], temp_nodes[1385], temp_nodes[1381], 64),
      (temp_nodes[818], my_string[247], jaxite_bool.constant(False, params), 1),
  ]
  outputs = jaxite_bool.pmap_lut3(inputs, sks, params)
  temp_nodes[1325] = outputs[0]
  temp_nodes[1345] = outputs[1]
  temp_nodes[1380] = outputs[2]
  temp_nodes[1387] = outputs[3]
  inputs = [
      (my_string[175], temp_nodes[65], temp_nodes[944], 7),
      (temp_nodes[1413], temp_nodes[1417], temp_nodes[1418], 128),
      (temp_nodes[1219], temp_nodes[1570], temp_nodes[1236], 64),
  ]
  outputs = jaxite_bool.pmap_lut3(inputs, sks, params)
  temp_nodes[1389] = outputs[0]
  temp_nodes[1412] = outputs[1]
  temp_nodes[1571] = outputs[2]
  inputs = [
      (
          temp_nodes[89],
          temp_nodes[1105],
          jaxite_bool.constant(False, params),
          4,
      ),
      (temp_nodes[1276], temp_nodes[1279], temp_nodes[1281], 64),
      (
          temp_nodes[1250],
          temp_nodes[1121],
          jaxite_bool.constant(False, params),
          8,
      ),
      (temp_nodes[1225], temp_nodes[1214], temp_nodes[559], 208),
  ]
  outputs = jaxite_bool.pmap_lut3(inputs, sks, params)
  temp_nodes[1104] = outputs[0]
  temp_nodes[1275] = outputs[1]
  temp_nodes[1290] = outputs[2]
  temp_nodes[1291] = outputs[3]
  inputs = [
      (temp_nodes[1293], temp_nodes[1214], temp_nodes[460], 208),
      (temp_nodes[1214], temp_nodes[1172], temp_nodes[1298], 16),
      (temp_nodes[261], my_string[197], jaxite_bool.constant(False, params), 8),
      (
          temp_nodes[1214],
          temp_nodes[1298],
          jaxite_bool.constant(False, params),
          4,
      ),
  ]
  outputs = jaxite_bool.pmap_lut3(inputs, sks, params)
  temp_nodes[1292] = outputs[0]
  temp_nodes[1297] = outputs[1]
  temp_nodes[1301] = outputs[2]
  temp_nodes[1313] = outputs[3]
  inputs = [
      (temp_nodes[1249], temp_nodes[1221], temp_nodes[1316], 16),
      (temp_nodes[1214], temp_nodes[1172], temp_nodes[1198], 1),
      (temp_nodes[1325], temp_nodes[1273], temp_nodes[1345], 128),
      (
          temp_nodes[1214],
          temp_nodes[1172],
          jaxite_bool.constant(False, params),
          1,
      ),
  ]
  outputs = jaxite_bool.pmap_lut3(inputs, sks, params)
  temp_nodes[1315] = outputs[0]
  temp_nodes[1322] = outputs[1]
  temp_nodes[1324] = outputs[2]
  temp_nodes[1348] = outputs[3]
  inputs = [
      (temp_nodes[1387], temp_nodes[1380], temp_nodes[1287], 64),
      (temp_nodes[1285], temp_nodes[1013], temp_nodes[1389], 128),
      (temp_nodes[1387], temp_nodes[1343], temp_nodes[1193], 64),
      (temp_nodes[1412], temp_nodes[941], temp_nodes[1389], 128),
  ]
  outputs = jaxite_bool.pmap_lut3(inputs, sks, params)
  temp_nodes[1379] = outputs[0]
  temp_nodes[1388] = outputs[1]
  temp_nodes[1409] = outputs[2]
  temp_nodes[1411] = outputs[3]
  inputs = [
      (
          temp_nodes[1571],
          temp_nodes[1266],
          jaxite_bool.constant(False, params),
          8,
      ),
  ]
  outputs = jaxite_bool.pmap_lut3(inputs, sks, params)
  temp_nodes[1572] = outputs[0]
  inputs = [
      (temp_nodes[134], temp_nodes[1104], temp_nodes[1970], 64),
      (temp_nodes[925], temp_nodes[1152], temp_nodes[1275], 64),
      (my_string[193], temp_nodes[1241], temp_nodes[1297], 64),
      (temp_nodes[1297], temp_nodes[1241], my_string[192], 128),
  ]
  outputs = jaxite_bool.pmap_lut3(inputs, sks, params)
  temp_nodes[1103] = outputs[0]
  temp_nodes[1274] = outputs[1]
  temp_nodes[1296] = outputs[2]
  temp_nodes[1299] = outputs[3]
  inputs = [
      (temp_nodes[994], temp_nodes[1301], my_string[194], 64),
      (
          temp_nodes[1572],
          temp_nodes[1576],
          jaxite_bool.constant(False, params),
          1,
      ),
      (
          temp_nodes[1292],
          temp_nodes[1313],
          jaxite_bool.constant(False, params),
          4,
      ),
      (temp_nodes[1315], temp_nodes[1290], temp_nodes[1273], 128),
  ]
  outputs = jaxite_bool.pmap_lut3(inputs, sks, params)
  temp_nodes[1300] = outputs[0]
  temp_nodes[1302] = outputs[1]
  temp_nodes[1312] = outputs[2]
  temp_nodes[1314] = outputs[3]
  inputs = [
      (
          temp_nodes[1322],
          temp_nodes[1127],
          jaxite_bool.constant(False, params),
          8,
      ),
      (temp_nodes[1291], temp_nodes[1292], temp_nodes[1324], 16),
      (
          temp_nodes[1219],
          temp_nodes[1348],
          jaxite_bool.constant(False, params),
          4,
      ),
      (
          temp_nodes[1572],
          temp_nodes[1267],
          jaxite_bool.constant(False, params),
          14,
      ),
  ]
  outputs = jaxite_bool.pmap_lut3(inputs, sks, params)
  temp_nodes[1321] = outputs[0]
  temp_nodes[1323] = outputs[1]
  temp_nodes[1347] = outputs[2]
  temp_nodes[1349] = outputs[3]
  inputs = [
      (
          temp_nodes[973],
          temp_nodes[1290],
          jaxite_bool.constant(False, params),
          4,
      ),
      (temp_nodes[935], my_string[244], my_string[245], 128),
      (temp_nodes[1134], temp_nodes[1379], temp_nodes[1388], 64),
      (temp_nodes[947], temp_nodes[1194], temp_nodes[1409], 128),
  ]
  outputs = jaxite_bool.pmap_lut3(inputs, sks, params)
  temp_nodes[1350] = outputs[0]
  temp_nodes[1376] = outputs[1]
  temp_nodes[1378] = outputs[2]
  temp_nodes[1408] = outputs[3]
  inputs = [
      (
          temp_nodes[1508],
          temp_nodes[1411],
          jaxite_bool.constant(False, params),
          4,
      ),
      (temp_nodes[889], my_string[252], my_string[253], 128),
  ]
  outputs = jaxite_bool.pmap_lut3(inputs, sks, params)
  temp_nodes[1410] = outputs[0]
  temp_nodes[1419] = outputs[1]
  inputs = [
      (temp_nodes[163], temp_nodes[140], temp_nodes[1103], 128),
      (temp_nodes[1273], temp_nodes[1274], temp_nodes[947], 128),
      (temp_nodes[1296], temp_nodes[1299], temp_nodes[1300], 128),
      (temp_nodes[1291], temp_nodes[1312], temp_nodes[1314], 64),
  ]
  outputs = jaxite_bool.pmap_lut3(inputs, sks, params)
  temp_nodes[1102] = outputs[0]
  temp_nodes[1272] = outputs[1]
  temp_nodes[1310] = outputs[2]
  temp_nodes[1311] = outputs[3]
  inputs = [
      (
          temp_nodes[1321],
          temp_nodes[1323],
          jaxite_bool.constant(False, params),
          4,
      ),
      (temp_nodes[1349], temp_nodes[1347], temp_nodes[1350], 64),
      (
          temp_nodes[1302],
          temp_nodes[997],
          jaxite_bool.constant(False, params),
          8,
      ),
      (temp_nodes[536], temp_nodes[1328], my_string[208], 64),
  ]
  outputs = jaxite_bool.pmap_lut3(inputs, sks, params)
  temp_nodes[1320] = outputs[0]
  temp_nodes[1346] = outputs[1]
  temp_nodes[1363] = outputs[2]
  temp_nodes[1364] = outputs[3]
  inputs = [
      (
          temp_nodes[1376],
          my_string[246],
          jaxite_bool.constant(False, params),
          1,
      ),
      (temp_nodes[1342], temp_nodes[1016], temp_nodes[1378], 128),
      (
          temp_nodes[1214],
          temp_nodes[935],
          jaxite_bool.constant(False, params),
          4,
      ),
      (temp_nodes[891], temp_nodes[1550], my_string[254], 13),
  ]
  outputs = jaxite_bool.pmap_lut3(inputs, sks, params)
  temp_nodes[1375] = outputs[0]
  temp_nodes[1377] = outputs[1]
  temp_nodes[1402] = outputs[2]
  temp_nodes[1406] = outputs[3]
  inputs = [
      (temp_nodes[1419], temp_nodes[1410], temp_nodes[1408], 64),
  ]
  outputs = jaxite_bool.pmap_lut3(inputs, sks, params)
  temp_nodes[1407] = outputs[0]
  inputs = [
      (
          temp_nodes[590],
          temp_nodes[598],
          jaxite_bool.constant(False, params),
          8,
      ),
      (temp_nodes[594], temp_nodes[1102], temp_nodes[592], 64),
      (temp_nodes[1221], temp_nodes[1290], temp_nodes[1272], 64),
      (
          temp_nodes[1310],
          temp_nodes[1311],
          jaxite_bool.constant(False, params),
          8,
      ),
  ]
  outputs = jaxite_bool.pmap_lut3(inputs, sks, params)
  temp_nodes[1100] = outputs[0]
  temp_nodes[1101] = outputs[1]
  temp_nodes[1271] = outputs[2]
  temp_nodes[1309] = outputs[3]
  inputs = [
      (
          temp_nodes[1320],
          temp_nodes[1346],
          jaxite_bool.constant(False, params),
          8,
      ),
      (
          temp_nodes[1363],
          temp_nodes[1364],
          jaxite_bool.constant(False, params),
          8,
      ),
      (temp_nodes[1311], temp_nodes[1310], temp_nodes[1321], 112),
      (temp_nodes[1311], temp_nodes[1310], temp_nodes[1214], 7),
  ]
  outputs = jaxite_bool.pmap_lut3(inputs, sks, params)
  temp_nodes[1319] = outputs[0]
  temp_nodes[1362] = outputs[1]
  temp_nodes[1368] = outputs[2]
  temp_nodes[1369] = outputs[3]
  inputs = [
      (
          temp_nodes[1349],
          temp_nodes[1347],
          jaxite_bool.constant(False, params),
          4,
      ),
      (temp_nodes[1375], temp_nodes[1236], temp_nodes[1377], 64),
      (temp_nodes[1311], temp_nodes[1310], temp_nodes[1402], 112),
      (temp_nodes[1375], temp_nodes[1406], temp_nodes[1407], 16),
  ]
  outputs = jaxite_bool.pmap_lut3(inputs, sks, params)
  temp_nodes[1373] = outputs[0]
  temp_nodes[1374] = outputs[1]
  temp_nodes[1401] = outputs[2]
  temp_nodes[1405] = outputs[3]
  inputs = [
      (temp_nodes[1346], temp_nodes[1320], temp_nodes[1187], 112),
      (
          temp_nodes[1348],
          temp_nodes[1175],
          jaxite_bool.constant(False, params),
          8,
      ),
  ]
  outputs = jaxite_bool.pmap_lut3(inputs, sks, params)
  temp_nodes[1423] = outputs[0]
  temp_nodes[1427] = outputs[1]
  inputs = [
      (temp_nodes[599], temp_nodes[1101], temp_nodes[1100], 64),
      (temp_nodes[1572], temp_nodes[1249], temp_nodes[1271], 16),
      (temp_nodes[1309], temp_nodes[1319], temp_nodes[1250], 16),
      (temp_nodes[1309], temp_nodes[1319], temp_nodes[1362], 16),
  ]
  outputs = jaxite_bool.pmap_lut3(inputs, sks, params)
  temp_nodes[1099] = outputs[0]
  temp_nodes[1255] = outputs[1]
  temp_nodes[1358] = outputs[2]
  temp_nodes[1361] = outputs[3]
  inputs = [
      (my_string[238], temp_nodes[1332], temp_nodes[1319], 14),
      (my_string[205], temp_nodes[1368], temp_nodes[409], 112),
      (temp_nodes[1309], temp_nodes[1319], temp_nodes[1277], 16),
      (temp_nodes[1291], temp_nodes[1373], temp_nodes[1374], 64),
  ]
  outputs = jaxite_bool.pmap_lut3(inputs, sks, params)
  temp_nodes[1365] = outputs[0]
  temp_nodes[1367] = outputs[1]
  temp_nodes[1371] = outputs[2]
  temp_nodes[1372] = outputs[3]
  inputs = [
      (temp_nodes[1319], temp_nodes[1401], temp_nodes[936], 64),
      (
          temp_nodes[1273],
          temp_nodes[1405],
          jaxite_bool.constant(False, params),
          8,
      ),
      (temp_nodes[1172], temp_nodes[1423], temp_nodes[1369], 64),
      (temp_nodes[1319], temp_nodes[936], my_string[216], 64),
  ]
  outputs = jaxite_bool.pmap_lut3(inputs, sks, params)
  temp_nodes[1400] = outputs[0]
  temp_nodes[1404] = outputs[1]
  temp_nodes[1422] = outputs[2]
  temp_nodes[1425] = outputs[3]
  inputs = [
      (temp_nodes[1427], temp_nodes[1319], temp_nodes[63], 208),
  ]
  outputs = jaxite_bool.pmap_lut3(inputs, sks, params)
  temp_nodes[1426] = outputs[0]
  inputs = [
      (
          temp_nodes[670],
          temp_nodes[1595],
          jaxite_bool.constant(False, params),
          1,
      ),
      (
          temp_nodes[1099],
          temp_nodes[740],
          jaxite_bool.constant(False, params),
          4,
      ),
      (
          temp_nodes[519],
          temp_nodes[585],
          jaxite_bool.constant(False, params),
          1,
      ),
      (temp_nodes[1291], temp_nodes[1292], temp_nodes[1255], 16),
  ]
  outputs = jaxite_bool.pmap_lut3(inputs, sks, params)
  temp_nodes[916] = outputs[0]
  temp_nodes[1098] = outputs[1]
  temp_nodes[1169] = outputs[2]
  temp_nodes[1254] = outputs[3]
  inputs = [
      (
          temp_nodes[1572],
          temp_nodes[1218],
          jaxite_bool.constant(False, params),
          4,
      ),
      (temp_nodes[1083], temp_nodes[1358], temp_nodes[1030], 64),
      (
          temp_nodes[1361],
          temp_nodes[1365],
          jaxite_bool.constant(False, params),
          8,
      ),
      (temp_nodes[1367], temp_nodes[1217], temp_nodes[1369], 16),
  ]
  outputs = jaxite_bool.pmap_lut3(inputs, sks, params)
  temp_nodes[1294] = outputs[0]
  temp_nodes[1357] = outputs[1]
  temp_nodes[1360] = outputs[2]
  temp_nodes[1366] = outputs[3]
  inputs = [
      (temp_nodes[1278], temp_nodes[1371], temp_nodes[1372], 208),
      (
          temp_nodes[1371],
          temp_nodes[1278],
          jaxite_bool.constant(False, params),
          4,
      ),
      (
          temp_nodes[1400],
          my_string[220],
          jaxite_bool.constant(False, params),
          8,
      ),
      (temp_nodes[1365], temp_nodes[1310], temp_nodes[1404], 128),
  ]
  outputs = jaxite_bool.pmap_lut3(inputs, sks, params)
  temp_nodes[1370] = outputs[0]
  temp_nodes[1398] = outputs[1]
  temp_nodes[1399] = outputs[2]
  temp_nodes[1403] = outputs[3]
  inputs = [
      (temp_nodes[461], temp_nodes[1422], temp_nodes[460], 112),
      (temp_nodes[1426], temp_nodes[1425], temp_nodes[1400], 64),
  ]
  outputs = jaxite_bool.pmap_lut3(inputs, sks, params)
  temp_nodes[1421] = outputs[0]
  temp_nodes[1424] = outputs[1]
  inputs = [
      (temp_nodes[247], temp_nodes[1938], temp_nodes[1098], 16),
      (
          temp_nodes[582],
          temp_nodes[1169],
          jaxite_bool.constant(False, params),
          4,
      ),
      (
          temp_nodes[1598],
          temp_nodes[916],
          jaxite_bool.constant(False, params),
          4,
      ),
      (
          temp_nodes[381],
          temp_nodes[421],
          jaxite_bool.constant(False, params),
          8,
      ),
  ]
  outputs = jaxite_bool.pmap_lut3(inputs, sks, params)
  temp_nodes[1097] = outputs[0]
  temp_nodes[1168] = outputs[1]
  temp_nodes[1211] = outputs[2]
  temp_nodes[1212] = outputs[3]
  inputs = [
      (temp_nodes[1082], temp_nodes[1294], temp_nodes[1254], 64),
      (temp_nodes[1296], temp_nodes[1299], temp_nodes[1300], 128),
      (
          temp_nodes[1309],
          temp_nodes[1319],
          jaxite_bool.constant(False, params),
          1,
      ),
      (my_string[165], temp_nodes[1357], temp_nodes[1841], 112),
  ]
  outputs = jaxite_bool.pmap_lut3(inputs, sks, params)
  temp_nodes[1253] = outputs[0]
  temp_nodes[1295] = outputs[1]
  temp_nodes[1308] = outputs[2]
  temp_nodes[1356] = outputs[3]
  inputs = [
      (temp_nodes[1360], temp_nodes[1366], temp_nodes[1370], 128),
      (
          temp_nodes[1142],
          temp_nodes[717],
          jaxite_bool.constant(False, params),
          1,
      ),
      (temp_nodes[807], temp_nodes[963], temp_nodes[800], 64),
      (temp_nodes[1398], temp_nodes[1399], temp_nodes[1403], 16),
  ]
  outputs = jaxite_bool.pmap_lut3(inputs, sks, params)
  temp_nodes[1359] = outputs[0]
  temp_nodes[1394] = outputs[1]
  temp_nodes[1395] = outputs[2]
  temp_nodes[1397] = outputs[3]
  inputs = [
      (temp_nodes[1421], temp_nodes[1366], temp_nodes[1424], 64),
      (
          temp_nodes[288],
          temp_nodes[1522],
          jaxite_bool.constant(False, params),
          1,
      ),
  ]
  outputs = jaxite_bool.pmap_lut3(inputs, sks, params)
  temp_nodes[1420] = outputs[0]
  temp_nodes[1458] = outputs[1]
  inputs = [
      (temp_nodes[963], temp_nodes[1590], temp_nodes[516], 128),
      (temp_nodes[1097], temp_nodes[240], temp_nodes[481], 128),
      (temp_nodes[788], temp_nodes[1151], temp_nodes[1168], 64),
      (temp_nodes[756], temp_nodes[1211], temp_nodes[1212], 64),
  ]
  outputs = jaxite_bool.pmap_lut3(inputs, sks, params)
  temp_nodes[962] = outputs[0]
  temp_nodes[1096] = outputs[1]
  temp_nodes[1167] = outputs[2]
  temp_nodes[1210] = outputs[3]
  inputs = [
      (
          temp_nodes[1253],
          temp_nodes[1295],
          jaxite_bool.constant(False, params),
          8,
      ),
      (
          temp_nodes[864],
          temp_nodes[838],
          jaxite_bool.constant(False, params),
          4,
      ),
      (
          temp_nodes[714],
          temp_nodes[565],
          jaxite_bool.constant(False, params),
          8,
      ),
      (
          temp_nodes[1308],
          temp_nodes[1313],
          jaxite_bool.constant(False, params),
          8,
      ),
  ]
  outputs = jaxite_bool.pmap_lut3(inputs, sks, params)
  temp_nodes[1252] = outputs[0]
  temp_nodes[1304] = outputs[1]
  temp_nodes[1305] = outputs[2]
  temp_nodes[1307] = outputs[3]
  inputs = [
      (temp_nodes[633], temp_nodes[247], temp_nodes[650], 16),
      (
          temp_nodes[1356],
          temp_nodes[1359],
          jaxite_bool.constant(False, params),
          4,
      ),
      (temp_nodes[1359], temp_nodes[1356], temp_nodes[1319], 13),
      (temp_nodes[918], temp_nodes[1394], temp_nodes[1395], 64),
  ]
  outputs = jaxite_bool.pmap_lut3(inputs, sks, params)
  temp_nodes[1353] = outputs[0]
  temp_nodes[1355] = outputs[1]
  temp_nodes[1392] = outputs[2]
  temp_nodes[1393] = outputs[3]
  inputs = [
      (
          temp_nodes[1397],
          temp_nodes[1420],
          jaxite_bool.constant(False, params),
          8,
      ),
      (temp_nodes[1359], temp_nodes[1356], temp_nodes[1369], 208),
      (temp_nodes[879], temp_nodes[1394], temp_nodes[1235], 64),
      (
          temp_nodes[1576],
          temp_nodes[1294],
          jaxite_bool.constant(False, params),
          4,
      ),
  ]
  outputs = jaxite_bool.pmap_lut3(inputs, sks, params)
  temp_nodes[1396] = outputs[0]
  temp_nodes[1431] = outputs[1]
  temp_nodes[1432] = outputs[2]
  temp_nodes[1436] = outputs[3]
  inputs = [
      (my_string[195], temp_nodes[1244], temp_nodes[1000], 64),
      (temp_nodes[935], temp_nodes[1128], my_string[202], 128),
      (
          temp_nodes[1308],
          temp_nodes[1242],
          jaxite_bool.constant(False, params),
          8,
      ),
      (temp_nodes[351], temp_nodes[367], temp_nodes[13], 1),
  ]
  outputs = jaxite_bool.pmap_lut3(inputs, sks, params)
  temp_nodes[1438] = outputs[0]
  temp_nodes[1442] = outputs[1]
  temp_nodes[1448] = outputs[2]
  temp_nodes[1452] = outputs[3]
  inputs = [
      (
          temp_nodes[327],
          temp_nodes[287],
          jaxite_bool.constant(False, params),
          4,
      ),
      (temp_nodes[1032], temp_nodes[1034], temp_nodes[1037], 1),
      (temp_nodes[327], temp_nodes[1458], temp_nodes[199], 64),
      (temp_nodes[1873], temp_nodes[219], temp_nodes[1884], 64),
  ]
  outputs = jaxite_bool.pmap_lut3(inputs, sks, params)
  temp_nodes[1455] = outputs[0]
  temp_nodes[1468] = outputs[1]
  temp_nodes[1470] = outputs[2]
  temp_nodes[1471] = outputs[3]
  inputs = [
      (temp_nodes[287], temp_nodes[326], temp_nodes[186], 128),
  ]
  outputs = jaxite_bool.pmap_lut3(inputs, sks, params)
  temp_nodes[1474] = outputs[0]
  inputs = [
      (temp_nodes[1582], temp_nodes[914], temp_nodes[916], 64),
      (
          temp_nodes[918],
          temp_nodes[962],
          jaxite_bool.constant(False, params),
          4,
      ),
      (
          temp_nodes[965],
          temp_nodes[1094],
          jaxite_bool.constant(False, params),
          4,
      ),
      (
          temp_nodes[582],
          temp_nodes[1096],
          jaxite_bool.constant(False, params),
          4,
      ),
  ]
  outputs = jaxite_bool.pmap_lut3(inputs, sks, params)
  temp_nodes[1581] = outputs[0]
  temp_nodes[917] = outputs[1]
  temp_nodes[964] = outputs[2]
  temp_nodes[1095] = outputs[3]
  inputs = [
      (
          temp_nodes[1109],
          temp_nodes[1167],
          jaxite_bool.constant(False, params),
          4,
      ),
      (
          temp_nodes[476],
          temp_nodes[238],
          jaxite_bool.constant(False, params),
          4,
      ),
      (temp_nodes[1172], temp_nodes[1204], temp_nodes[1210], 64),
      (temp_nodes[1214], temp_nodes[1234], temp_nodes[870], 64),
  ]
  outputs = jaxite_bool.pmap_lut3(inputs, sks, params)
  temp_nodes[1108] = outputs[0]
  temp_nodes[1170] = outputs[1]
  temp_nodes[1171] = outputs[2]
  temp_nodes[1213] = outputs[3]
  inputs = [
      (temp_nodes[1252], temp_nodes[1302], temp_nodes[1247], 64),
      (temp_nodes[1304], temp_nodes[1305], my_string[146], 128),
      (my_string[147], temp_nodes[1305], temp_nodes[1304], 64),
      (temp_nodes[1307], temp_nodes[963], temp_nodes[1588], 128),
  ]
  outputs = jaxite_bool.pmap_lut3(inputs, sks, params)
  temp_nodes[1251] = outputs[0]
  temp_nodes[1303] = outputs[1]
  temp_nodes[1306] = outputs[2]
  temp_nodes[1351] = outputs[3]
  inputs = [
      (temp_nodes[1353], temp_nodes[1163], my_string[154], 128),
      (temp_nodes[1353], temp_nodes[473], temp_nodes[240], 128),
      (
          temp_nodes[1358],
          temp_nodes[1020],
          jaxite_bool.constant(False, params),
          8,
      ),
      (temp_nodes[1392], temp_nodes[1348], temp_nodes[1393], 128),
  ]
  outputs = jaxite_bool.pmap_lut3(inputs, sks, params)
  temp_nodes[1352] = outputs[0]
  temp_nodes[1354] = outputs[1]
  temp_nodes[1390] = outputs[2]
  temp_nodes[1391] = outputs[3]
  inputs = [
      (
          temp_nodes[1392],
          temp_nodes[1348],
          jaxite_bool.constant(False, params),
          8,
      ),
      (temp_nodes[1396], my_string[169], temp_nodes[1393], 16),
      (temp_nodes[1396], temp_nodes[1431], temp_nodes[1432], 64),
      (temp_nodes[1198], temp_nodes[1431], temp_nodes[1206], 64),
  ]
  outputs = jaxite_bool.pmap_lut3(inputs, sks, params)
  temp_nodes[1428] = outputs[0]
  temp_nodes[1429] = outputs[1]
  temp_nodes[1430] = outputs[2]
  temp_nodes[1433] = outputs[3]
  inputs = [
      (temp_nodes[1358], temp_nodes[1239], temp_nodes[1240], 128),
      (
          temp_nodes[1436],
          temp_nodes[995],
          jaxite_bool.constant(False, params),
          8,
      ),
      (
          temp_nodes[1363],
          temp_nodes[1438],
          jaxite_bool.constant(False, params),
          8,
      ),
      (
          temp_nodes[1355],
          temp_nodes[1401],
          jaxite_bool.constant(False, params),
          4,
      ),
  ]
  outputs = jaxite_bool.pmap_lut3(inputs, sks, params)
  temp_nodes[1434] = outputs[0]
  temp_nodes[1435] = outputs[1]
  temp_nodes[1437] = outputs[2]
  temp_nodes[1439] = outputs[3]
  inputs = [
      (temp_nodes[1355], temp_nodes[1401], temp_nodes[1326], 64),
      (temp_nodes[1309], temp_nodes[1322], temp_nodes[1442], 64),
      (temp_nodes[1355], temp_nodes[1396], temp_nodes[935], 16),
      (my_string[203], temp_nodes[1327], temp_nodes[1322], 64),
  ]
  outputs = jaxite_bool.pmap_lut3(inputs, sks, params)
  temp_nodes[1440] = outputs[0]
  temp_nodes[1441] = outputs[1]
  temp_nodes[1443] = outputs[2]
  temp_nodes[1444] = outputs[3]
  inputs = [
      (
          temp_nodes[1396],
          temp_nodes[1422],
          jaxite_bool.constant(False, params),
          4,
      ),
      (
          temp_nodes[1355],
          temp_nodes[1400],
          jaxite_bool.constant(False, params),
          4,
      ),
      (temp_nodes[1355], temp_nodes[1396], temp_nodes[1448], 16),
      (temp_nodes[1396], temp_nodes[1392], temp_nodes[1243], 64),
  ]
  outputs = jaxite_bool.pmap_lut3(inputs, sks, params)
  temp_nodes[1445] = outputs[0]
  temp_nodes[1446] = outputs[1]
  temp_nodes[1447] = outputs[2]
  temp_nodes[1449] = outputs[3]
  inputs = [
      (
          temp_nodes[1611],
          temp_nodes[1905],
          jaxite_bool.constant(False, params),
          8,
      ),
      (temp_nodes[351], temp_nodes[94], temp_nodes[1900], 64),
      (
          temp_nodes[1452],
          temp_nodes[1899],
          jaxite_bool.constant(False, params),
          8,
      ),
      (temp_nodes[1455], temp_nodes[1894], my_string[32], 128),
  ]
  outputs = jaxite_bool.pmap_lut3(inputs, sks, params)
  temp_nodes[1450] = outputs[0]
  temp_nodes[1451] = outputs[1]
  temp_nodes[1453] = outputs[2]
  temp_nodes[1454] = outputs[3]
  inputs = [
      (temp_nodes[588], temp_nodes[1455], temp_nodes[1894], 64),
      (
          temp_nodes[326],
          temp_nodes[1944],
          jaxite_bool.constant(False, params),
          8,
      ),
      (my_string[49], temp_nodes[1943], temp_nodes[1098], 64),
      (temp_nodes[1458], temp_nodes[1943], my_string[50], 128),
  ]
  outputs = jaxite_bool.pmap_lut3(inputs, sks, params)
  temp_nodes[1456] = outputs[0]
  temp_nodes[1457] = outputs[1]
  temp_nodes[1459] = outputs[2]
  temp_nodes[1460] = outputs[3]
  inputs = [
      (my_string[51], temp_nodes[1862], temp_nodes[1608], 64),
      (
          temp_nodes[1168],
          temp_nodes[349],
          jaxite_bool.constant(False, params),
          8,
      ),
      (temp_nodes[1037], temp_nodes[741], temp_nodes[744], 1),
      (temp_nodes[1938], temp_nodes[1455], temp_nodes[1943], 64),
  ]
  outputs = jaxite_bool.pmap_lut3(inputs, sks, params)
  temp_nodes[1461] = outputs[0]
  temp_nodes[1462] = outputs[1]
  temp_nodes[1463] = outputs[2]
  temp_nodes[1464] = outputs[3]
  inputs = [
      (temp_nodes[281], temp_nodes[1611], temp_nodes[4], 128),
      (temp_nodes[1602], temp_nodes[136], temp_nodes[758], 16),
      (temp_nodes[744], temp_nodes[1099], temp_nodes[1468], 16),
      (
          temp_nodes[1470],
          temp_nodes[1471],
          jaxite_bool.constant(False, params),
          8,
      ),
  ]
  outputs = jaxite_bool.pmap_lut3(inputs, sks, params)
  temp_nodes[1465] = outputs[0]
  temp_nodes[1466] = outputs[1]
  temp_nodes[1467] = outputs[2]
  temp_nodes[1469] = outputs[3]
  inputs = [
      (temp_nodes[367], temp_nodes[1169], temp_nodes[480], 64),
      (
          temp_nodes[740],
          temp_nodes[1474],
          jaxite_bool.constant(False, params),
          8,
      ),
      (
          temp_nodes[952],
          temp_nodes[793],
          jaxite_bool.constant(False, params),
          8,
      ),
      (temp_nodes[1099], temp_nodes[337], temp_nodes[1458], 16),
  ]
  outputs = jaxite_bool.pmap_lut3(inputs, sks, params)
  temp_nodes[1472] = outputs[0]
  temp_nodes[1473] = outputs[1]
  temp_nodes[1475] = outputs[2]
  temp_nodes[1476] = outputs[3]
  inputs = [
      (temp_nodes[1828], temp_nodes[1615], my_string[90], 128),
      (my_string[91], temp_nodes[223], temp_nodes[1604], 64),
  ]
  outputs = jaxite_bool.pmap_lut3(inputs, sks, params)
  temp_nodes[1477] = outputs[0]
  temp_nodes[1478] = outputs[1]
  inputs = [
      (
          temp_nodes[1581],
          temp_nodes[284],
          jaxite_bool.constant(False, params),
          8,
      ),
      (
          temp_nodes[1581],
          temp_nodes[1599],
          jaxite_bool.constant(False, params),
          7,
      ),
      (temp_nodes[1581], temp_nodes[1600], my_string[98], 128),
      (my_string[99], temp_nodes[1600], temp_nodes[1581], 191),
  ]
  outputs = jaxite_bool.pmap_lut3(inputs, sks, params)
  out[96] = outputs[0]
  out[97] = outputs[1]
  out[98] = outputs[2]
  out[99] = outputs[3]
  inputs = [
      (temp_nodes[1581], temp_nodes[1601], temp_nodes[50], 128),
      (temp_nodes[917], my_string[104], jaxite_bool.constant(False, params), 8),
      (
          my_string[105],
          temp_nodes[917],
          jaxite_bool.constant(False, params),
          11,
      ),
      (temp_nodes[917], my_string[106], jaxite_bool.constant(False, params), 8),
  ]
  outputs = jaxite_bool.pmap_lut3(inputs, sks, params)
  out[100] = outputs[0]
  out[104] = outputs[1]
  out[105] = outputs[2]
  out[106] = outputs[3]
  inputs = [
      (
          my_string[107],
          temp_nodes[917],
          jaxite_bool.constant(False, params),
          11,
      ),
      (temp_nodes[917], my_string[108], jaxite_bool.constant(False, params), 8),
      (temp_nodes[1595], temp_nodes[964], temp_nodes[579], 64),
      (
          temp_nodes[964],
          temp_nodes[1594],
          jaxite_bool.constant(False, params),
          7,
      ),
  ]
  outputs = jaxite_bool.pmap_lut3(inputs, sks, params)
  out[107] = outputs[0]
  out[108] = outputs[1]
  out[112] = outputs[2]
  out[113] = outputs[3]
  inputs = [
      (temp_nodes[964], temp_nodes[1095], my_string[114], 128),
      (my_string[115], temp_nodes[1095], temp_nodes[964], 191),
      (temp_nodes[964], temp_nodes[1095], my_string[116], 128),
      (temp_nodes[1108], temp_nodes[1170], my_string[120], 128),
  ]
  outputs = jaxite_bool.pmap_lut3(inputs, sks, params)
  out[114] = outputs[0]
  out[115] = outputs[1]
  out[116] = outputs[2]
  out[120] = outputs[3]
  inputs = [
      (
          temp_nodes[1108],
          temp_nodes[666],
          jaxite_bool.constant(False, params),
          7,
      ),
      (temp_nodes[1108], temp_nodes[1170], my_string[122], 128),
      (my_string[123], temp_nodes[1170], temp_nodes[1108], 191),
      (
          temp_nodes[1108],
          temp_nodes[237],
          jaxite_bool.constant(False, params),
          8,
      ),
  ]
  outputs = jaxite_bool.pmap_lut3(inputs, sks, params)
  out[121] = outputs[0]
  out[122] = outputs[1]
  out[123] = outputs[2]
  out[124] = outputs[3]
  inputs = [
      (
          temp_nodes[1171],
          my_string[128],
          jaxite_bool.constant(False, params),
          8,
      ),
      (
          my_string[129],
          temp_nodes[1171],
          jaxite_bool.constant(False, params),
          11,
      ),
      (
          temp_nodes[1171],
          my_string[130],
          jaxite_bool.constant(False, params),
          8,
      ),
      (
          my_string[131],
          temp_nodes[1171],
          jaxite_bool.constant(False, params),
          11,
      ),
  ]
  outputs = jaxite_bool.pmap_lut3(inputs, sks, params)
  out[128] = outputs[0]
  out[129] = outputs[1]
  out[130] = outputs[2]
  out[131] = outputs[3]
  inputs = [
      (
          temp_nodes[1171],
          my_string[132],
          jaxite_bool.constant(False, params),
          8,
      ),
      (
          temp_nodes[1213],
          my_string[136],
          jaxite_bool.constant(False, params),
          8,
      ),
      (
          my_string[137],
          temp_nodes[1213],
          jaxite_bool.constant(False, params),
          11,
      ),
      (
          temp_nodes[1213],
          my_string[138],
          jaxite_bool.constant(False, params),
          8,
      ),
  ]
  outputs = jaxite_bool.pmap_lut3(inputs, sks, params)
  out[132] = outputs[0]
  out[136] = outputs[1]
  out[137] = outputs[2]
  out[138] = outputs[3]
  inputs = [
      (
          my_string[139],
          temp_nodes[1213],
          jaxite_bool.constant(False, params),
          11,
      ),
      (
          temp_nodes[1213],
          my_string[140],
          jaxite_bool.constant(False, params),
          8,
      ),
      (
          temp_nodes[1251],
          temp_nodes[1585],
          jaxite_bool.constant(False, params),
          8,
      ),
      (temp_nodes[864], temp_nodes[1251], temp_nodes[834], 191),
  ]
  outputs = jaxite_bool.pmap_lut3(inputs, sks, params)
  out[139] = outputs[0]
  out[140] = outputs[1]
  out[144] = outputs[2]
  out[145] = outputs[3]
  inputs = [
      (
          temp_nodes[1251],
          temp_nodes[1303],
          jaxite_bool.constant(False, params),
          8,
      ),
      (
          temp_nodes[1251],
          temp_nodes[1306],
          jaxite_bool.constant(False, params),
          7,
      ),
      (temp_nodes[1251], temp_nodes[971], temp_nodes[535], 128),
      (temp_nodes[1582], temp_nodes[1307], temp_nodes[1089], 64),
  ]
  outputs = jaxite_bool.pmap_lut3(inputs, sks, params)
  out[146] = outputs[0]
  out[147] = outputs[1]
  out[148] = outputs[2]
  out[152] = outputs[3]
  inputs = [
      (
          temp_nodes[1307],
          temp_nodes[950],
          jaxite_bool.constant(False, params),
          7,
      ),
      (temp_nodes[1351], temp_nodes[1352], temp_nodes[240], 128),
      (my_string[155], temp_nodes[1354], temp_nodes[1351], 191),
      (
          temp_nodes[1351],
          temp_nodes[631],
          jaxite_bool.constant(False, params),
          8,
      ),
  ]
  outputs = jaxite_bool.pmap_lut3(inputs, sks, params)
  out[153] = outputs[0]
  out[154] = outputs[1]
  out[155] = outputs[2]
  out[156] = outputs[3]
  inputs = [
      (temp_nodes[1355], temp_nodes[1358], temp_nodes[1019], 64),
      (temp_nodes[1355], temp_nodes[1358], temp_nodes[1025], 191),
      (temp_nodes[1355], temp_nodes[1390], my_string[162], 64),
      (temp_nodes[1355], my_string[163], temp_nodes[1390], 239),
  ]
  outputs = jaxite_bool.pmap_lut3(inputs, sks, params)
  out[160] = outputs[0]
  out[161] = outputs[1]
  out[162] = outputs[2]
  out[163] = outputs[3]
  inputs = [
      (
          temp_nodes[1355],
          temp_nodes[1357],
          jaxite_bool.constant(False, params),
          4,
      ),
      (temp_nodes[1396], temp_nodes[1391], my_string[168], 64),
      (
          temp_nodes[1428],
          temp_nodes[1429],
          jaxite_bool.constant(False, params),
          7,
      ),
      (temp_nodes[1396], temp_nodes[1391], my_string[170], 64),
  ]
  outputs = jaxite_bool.pmap_lut3(inputs, sks, params)
  out[164] = outputs[0]
  out[168] = outputs[1]
  out[169] = outputs[2]
  out[170] = outputs[3]
  inputs = [
      (temp_nodes[1396], my_string[171], temp_nodes[1391], 239),
      (temp_nodes[1396], temp_nodes[1428], temp_nodes[1176], 64),
      (
          temp_nodes[1430],
          temp_nodes[1264],
          jaxite_bool.constant(False, params),
          8,
      ),
      (
          temp_nodes[1430],
          temp_nodes[1207],
          jaxite_bool.constant(False, params),
          7,
      ),
  ]
  outputs = jaxite_bool.pmap_lut3(inputs, sks, params)
  out[171] = outputs[0]
  out[172] = outputs[1]
  out[176] = outputs[2]
  out[177] = outputs[3]
  inputs = [
      (temp_nodes[1430], temp_nodes[895], my_string[178], 128),
      (my_string[179], temp_nodes[895], temp_nodes[1430], 191),
      (
          temp_nodes[1396],
          temp_nodes[1433],
          jaxite_bool.constant(False, params),
          4,
      ),
      (temp_nodes[1396], temp_nodes[1434], my_string[184], 64),
  ]
  outputs = jaxite_bool.pmap_lut3(inputs, sks, params)
  out[178] = outputs[0]
  out[179] = outputs[1]
  out[180] = outputs[2]
  out[184] = outputs[3]
  inputs = [
      (temp_nodes[1396], temp_nodes[1308], temp_nodes[1237], 191),
      (temp_nodes[1396], temp_nodes[1434], my_string[186], 64),
      (temp_nodes[1396], my_string[187], temp_nodes[1434], 239),
      (temp_nodes[1396], temp_nodes[1358], temp_nodes[1246], 64),
  ]
  outputs = jaxite_bool.pmap_lut3(inputs, sks, params)
  out[185] = outputs[0]
  out[186] = outputs[1]
  out[187] = outputs[2]
  out[188] = outputs[3]
  inputs = [
      (
          temp_nodes[1392],
          temp_nodes[1299],
          jaxite_bool.constant(False, params),
          8,
      ),
      (
          temp_nodes[1392],
          temp_nodes[1296],
          jaxite_bool.constant(False, params),
          7,
      ),
      (temp_nodes[1392], temp_nodes[1435], my_string[194], 128),
      (temp_nodes[1392], temp_nodes[1437], temp_nodes[1218], 127),
  ]
  outputs = jaxite_bool.pmap_lut3(inputs, sks, params)
  out[192] = outputs[0]
  out[193] = outputs[1]
  out[194] = outputs[2]
  out[195] = outputs[3]
  inputs = [
      (temp_nodes[1392], temp_nodes[1436], temp_nodes[994], 128),
      (temp_nodes[1396], temp_nodes[1439], temp_nodes[1345], 64),
      (
          temp_nodes[1396],
          temp_nodes[1440],
          jaxite_bool.constant(False, params),
          11,
      ),
      (temp_nodes[1355], temp_nodes[1396], temp_nodes[1441], 16),
  ]
  outputs = jaxite_bool.pmap_lut3(inputs, sks, params)
  out[196] = outputs[0]
  out[200] = outputs[1]
  out[201] = outputs[2]
  out[202] = outputs[3]
  inputs = [
      (temp_nodes[1309], temp_nodes[1443], temp_nodes[1444], 191),
      (temp_nodes[1355], temp_nodes[1396], temp_nodes[1368], 16),
      (
          temp_nodes[1396],
          temp_nodes[1361],
          jaxite_bool.constant(False, params),
          4,
      ),
      (
          my_string[209],
          temp_nodes[1445],
          jaxite_bool.constant(False, params),
          11,
      ),
  ]
  outputs = jaxite_bool.pmap_lut3(inputs, sks, params)
  out[203] = outputs[0]
  out[204] = outputs[1]
  out[208] = outputs[2]
  out[209] = outputs[3]
  inputs = [
      (
          temp_nodes[1445],
          my_string[210],
          jaxite_bool.constant(False, params),
          8,
      ),
      (
          my_string[211],
          temp_nodes[1445],
          jaxite_bool.constant(False, params),
          11,
      ),
      (
          temp_nodes[1445],
          my_string[212],
          jaxite_bool.constant(False, params),
          8,
      ),
      (
          temp_nodes[1439],
          temp_nodes[1425],
          jaxite_bool.constant(False, params),
          8,
      ),
  ]
  outputs = jaxite_bool.pmap_lut3(inputs, sks, params)
  out[210] = outputs[0]
  out[211] = outputs[1]
  out[212] = outputs[2]
  out[216] = outputs[3]
  inputs = [
      (
          my_string[217],
          temp_nodes[1446],
          jaxite_bool.constant(False, params),
          11,
      ),
      (
          temp_nodes[1446],
          my_string[218],
          jaxite_bool.constant(False, params),
          8,
      ),
      (
          my_string[219],
          temp_nodes[1446],
          jaxite_bool.constant(False, params),
          11,
      ),
      (
          temp_nodes[1355],
          temp_nodes[1399],
          jaxite_bool.constant(False, params),
          4,
      ),
  ]
  outputs = jaxite_bool.pmap_lut3(inputs, sks, params)
  out[217] = outputs[0]
  out[218] = outputs[1]
  out[219] = outputs[2]
  out[220] = outputs[3]
  inputs = [
      (
          temp_nodes[1447],
          my_string[224],
          jaxite_bool.constant(False, params),
          8,
      ),
      (
          my_string[225],
          temp_nodes[1447],
          jaxite_bool.constant(False, params),
          11,
      ),
      (
          temp_nodes[1447],
          my_string[226],
          jaxite_bool.constant(False, params),
          8,
      ),
      (
          my_string[227],
          temp_nodes[1447],
          jaxite_bool.constant(False, params),
          11,
      ),
  ]
  outputs = jaxite_bool.pmap_lut3(inputs, sks, params)
  out[224] = outputs[0]
  out[225] = outputs[1]
  out[226] = outputs[2]
  out[227] = outputs[3]
  inputs = [
      (
          temp_nodes[1447],
          my_string[228],
          jaxite_bool.constant(False, params),
          8,
      ),
      (
          temp_nodes[1449],
          my_string[232],
          jaxite_bool.constant(False, params),
          8,
      ),
      (
          my_string[233],
          temp_nodes[1449],
          jaxite_bool.constant(False, params),
          11,
      ),
      (
          temp_nodes[1449],
          my_string[234],
          jaxite_bool.constant(False, params),
          8,
      ),
  ]
  outputs = jaxite_bool.pmap_lut3(inputs, sks, params)
  out[228] = outputs[0]
  out[232] = outputs[1]
  out[233] = outputs[2]
  out[234] = outputs[3]
  inputs = [
      (
          my_string[235],
          temp_nodes[1449],
          jaxite_bool.constant(False, params),
          11,
      ),
      (
          temp_nodes[1449],
          my_string[236],
          jaxite_bool.constant(False, params),
          8,
      ),
      (
          temp_nodes[1443],
          my_string[240],
          jaxite_bool.constant(False, params),
          8,
      ),
      (
          my_string[241],
          temp_nodes[1443],
          jaxite_bool.constant(False, params),
          11,
      ),
  ]
  outputs = jaxite_bool.pmap_lut3(inputs, sks, params)
  out[235] = outputs[0]
  out[236] = outputs[1]
  out[240] = outputs[2]
  out[241] = outputs[3]
  inputs = [
      (
          temp_nodes[1443],
          my_string[242],
          jaxite_bool.constant(False, params),
          8,
      ),
      (
          my_string[243],
          temp_nodes[1443],
          jaxite_bool.constant(False, params),
          11,
      ),
      (
          temp_nodes[1443],
          my_string[244],
          jaxite_bool.constant(False, params),
          8,
      ),
      (temp_nodes[1396], temp_nodes[1550], my_string[248], 16),
  ]
  outputs = jaxite_bool.pmap_lut3(inputs, sks, params)
  out[242] = outputs[0]
  out[243] = outputs[1]
  out[244] = outputs[2]
  out[248] = outputs[3]
  inputs = [
      (temp_nodes[1396], temp_nodes[1550], my_string[249], 254),
      (temp_nodes[1396], temp_nodes[1550], my_string[250], 16),
      (temp_nodes[1396], temp_nodes[1550], my_string[251], 254),
      (temp_nodes[1396], temp_nodes[1550], my_string[252], 16),
  ]
  outputs = jaxite_bool.pmap_lut3(inputs, sks, params)
  out[249] = outputs[0]
  out[250] = outputs[1]
  out[251] = outputs[2]
  out[252] = outputs[3]
  inputs = [
      (temp_nodes[327], temp_nodes[1867], my_string[0], 16),
      (temp_nodes[327], temp_nodes[1867], my_string[1], 254),
      (temp_nodes[327], temp_nodes[1867], my_string[2], 16),
      (temp_nodes[327], temp_nodes[1867], my_string[3], 254),
  ]
  outputs = jaxite_bool.pmap_lut3(inputs, sks, params)
  out[0] = outputs[0]
  out[1] = outputs[1]
  out[2] = outputs[2]
  out[3] = outputs[3]
  inputs = [
      (temp_nodes[327], temp_nodes[1867], my_string[4], 16),
      (temp_nodes[1450], my_string[8], jaxite_bool.constant(False, params), 8),
      (my_string[9], temp_nodes[1450], jaxite_bool.constant(False, params), 11),
      (temp_nodes[1450], my_string[10], jaxite_bool.constant(False, params), 8),
  ]
  outputs = jaxite_bool.pmap_lut3(inputs, sks, params)
  out[4] = outputs[0]
  out[8] = outputs[1]
  out[9] = outputs[2]
  out[10] = outputs[3]
  inputs = [
      (
          my_string[11],
          temp_nodes[1450],
          jaxite_bool.constant(False, params),
          11,
      ),
      (temp_nodes[1450], my_string[12], jaxite_bool.constant(False, params), 8),
      (temp_nodes[1451], my_string[16], jaxite_bool.constant(False, params), 8),
      (
          my_string[17],
          temp_nodes[1451],
          jaxite_bool.constant(False, params),
          11,
      ),
  ]
  outputs = jaxite_bool.pmap_lut3(inputs, sks, params)
  out[11] = outputs[0]
  out[12] = outputs[1]
  out[16] = outputs[2]
  out[17] = outputs[3]
  inputs = [
      (temp_nodes[1451], my_string[18], jaxite_bool.constant(False, params), 8),
      (
          my_string[19],
          temp_nodes[1451],
          jaxite_bool.constant(False, params),
          11,
      ),
      (temp_nodes[1451], my_string[20], jaxite_bool.constant(False, params), 8),
      (temp_nodes[1452], temp_nodes[1899], my_string[24], 128),
  ]
  outputs = jaxite_bool.pmap_lut3(inputs, sks, params)
  out[18] = outputs[0]
  out[19] = outputs[1]
  out[20] = outputs[2]
  out[24] = outputs[3]
  inputs = [
      (my_string[25], temp_nodes[1899], temp_nodes[1452], 191),
      (temp_nodes[1452], temp_nodes[1899], my_string[26], 128),
      (
          my_string[27],
          temp_nodes[1453],
          jaxite_bool.constant(False, params),
          11,
      ),
      (temp_nodes[1453], my_string[28], jaxite_bool.constant(False, params), 8),
  ]
  outputs = jaxite_bool.pmap_lut3(inputs, sks, params)
  out[25] = outputs[0]
  out[26] = outputs[1]
  out[27] = outputs[2]
  out[28] = outputs[3]
  inputs = [
      (
          temp_nodes[1099],
          temp_nodes[1454],
          jaxite_bool.constant(False, params),
          4,
      ),
      (
          my_string[33],
          temp_nodes[1456],
          jaxite_bool.constant(False, params),
          11,
      ),
      (temp_nodes[1456], my_string[34], jaxite_bool.constant(False, params), 8),
      (
          my_string[35],
          temp_nodes[1456],
          jaxite_bool.constant(False, params),
          11,
      ),
  ]
  outputs = jaxite_bool.pmap_lut3(inputs, sks, params)
  out[32] = outputs[0]
  out[33] = outputs[1]
  out[34] = outputs[2]
  out[35] = outputs[3]
  inputs = [
      (temp_nodes[1456], my_string[36], jaxite_bool.constant(False, params), 8),
      (temp_nodes[497], temp_nodes[46], jaxite_bool.constant(False, params), 8),
      (temp_nodes[497], temp_nodes[12], jaxite_bool.constant(False, params), 7),
      (temp_nodes[584], temp_nodes[1457], my_string[42], 128),
  ]
  outputs = jaxite_bool.pmap_lut3(inputs, sks, params)
  out[36] = outputs[0]
  out[40] = outputs[1]
  out[41] = outputs[2]
  out[42] = outputs[3]
  inputs = [
      (my_string[43], temp_nodes[1457], temp_nodes[584], 191),
      (
          temp_nodes[497],
          temp_nodes[325],
          jaxite_bool.constant(False, params),
          8,
      ),
      (temp_nodes[1098], temp_nodes[1458], temp_nodes[1960], 128),
      (
          temp_nodes[1459],
          temp_nodes[1458],
          jaxite_bool.constant(False, params),
          7,
      ),
  ]
  outputs = jaxite_bool.pmap_lut3(inputs, sks, params)
  out[43] = outputs[0]
  out[44] = outputs[1]
  out[48] = outputs[2]
  out[49] = outputs[3]
  inputs = [
      (
          temp_nodes[1098],
          temp_nodes[1460],
          jaxite_bool.constant(False, params),
          8,
      ),
      (
          temp_nodes[1098],
          temp_nodes[1461],
          jaxite_bool.constant(False, params),
          7,
      ),
      (
          temp_nodes[1598],
          temp_nodes[523],
          jaxite_bool.constant(False, params),
          4,
      ),
      (temp_nodes[1462], temp_nodes[1954], temp_nodes[1948], 128),
  ]
  outputs = jaxite_bool.pmap_lut3(inputs, sks, params)
  out[50] = outputs[0]
  out[51] = outputs[1]
  out[52] = outputs[2]
  out[56] = outputs[3]
  inputs = [
      (
          temp_nodes[1462],
          temp_nodes[1937],
          jaxite_bool.constant(False, params),
          7,
      ),
      (temp_nodes[1463], temp_nodes[1464], my_string[58], 128),
      (my_string[59], temp_nodes[1464], temp_nodes[1463], 191),
      (
          temp_nodes[1462],
          temp_nodes[1946],
          jaxite_bool.constant(False, params),
          8,
      ),
  ]
  outputs = jaxite_bool.pmap_lut3(inputs, sks, params)
  out[57] = outputs[0]
  out[58] = outputs[1]
  out[59] = outputs[2]
  out[60] = outputs[3]
  inputs = [
      (temp_nodes[1211], temp_nodes[281], temp_nodes[1610], 128),
      (my_string[65], temp_nodes[1465], temp_nodes[1211], 191),
      (temp_nodes[1211], temp_nodes[1465], my_string[66], 128),
      (my_string[67], temp_nodes[1465], temp_nodes[1211], 191),
  ]
  outputs = jaxite_bool.pmap_lut3(inputs, sks, params)
  out[64] = outputs[0]
  out[65] = outputs[1]
  out[66] = outputs[2]
  out[67] = outputs[3]
  inputs = [
      (temp_nodes[1211], temp_nodes[281], temp_nodes[6], 128),
      (temp_nodes[1466], temp_nodes[1607], my_string[72], 128),
      (my_string[73], temp_nodes[1469], temp_nodes[1467], 191),
      (temp_nodes[1467], temp_nodes[1469], my_string[74], 128),
  ]
  outputs = jaxite_bool.pmap_lut3(inputs, sks, params)
  out[68] = outputs[0]
  out[72] = outputs[1]
  out[73] = outputs[2]
  out[74] = outputs[3]
  inputs = [
      (my_string[75], temp_nodes[1607], temp_nodes[1467], 191),
      (
          temp_nodes[1466],
          temp_nodes[227],
          jaxite_bool.constant(False, params),
          8,
      ),
      (temp_nodes[768], temp_nodes[514], my_string[80], 128),
      (
          temp_nodes[838],
          temp_nodes[1472],
          jaxite_bool.constant(False, params),
          7,
      ),
  ]
  outputs = jaxite_bool.pmap_lut3(inputs, sks, params)
  out[75] = outputs[0]
  out[76] = outputs[1]
  out[80] = outputs[2]
  out[81] = outputs[3]
  inputs = [
      (temp_nodes[838], temp_nodes[1473], my_string[82], 128),
      (my_string[83], temp_nodes[1473], temp_nodes[838], 191),
      (temp_nodes[838], temp_nodes[514], my_string[84], 128),
      (temp_nodes[1475], temp_nodes[1476], temp_nodes[209], 128),
  ]
  outputs = jaxite_bool.pmap_lut3(inputs, sks, params)
  out[82] = outputs[0]
  out[83] = outputs[1]
  out[84] = outputs[2]
  out[88] = outputs[3]
  inputs = [
      (temp_nodes[1475], temp_nodes[1476], temp_nodes[222], 127),
      (temp_nodes[1475], temp_nodes[1604], temp_nodes[1477], 128),
      (
          temp_nodes[1475],
          temp_nodes[1478],
          jaxite_bool.constant(False, params),
          7,
      ),
      (temp_nodes[1475], temp_nodes[1476], temp_nodes[220], 128),
  ]
  outputs = jaxite_bool.pmap_lut3(inputs, sks, params)
  out[89] = outputs[0]
  out[90] = outputs[1]
  out[91] = outputs[2]
  out[92] = outputs[3]
  out[101] = my_string[101]
  out[102] = my_string[102]
  out[103] = my_string[103]
  out[109] = my_string[109]
  out[110] = my_string[110]
  out[111] = my_string[111]
  out[117] = my_string[117]
  out[118] = my_string[118]
  out[119] = my_string[119]
  out[125] = my_string[125]
  out[126] = my_string[126]
  out[127] = my_string[127]
  out[133] = my_string[133]
  out[134] = my_string[134]
  out[135] = my_string[135]
  out[13] = my_string[13]
  out[141] = my_string[141]
  out[142] = my_string[142]
  out[143] = my_string[143]
  out[149] = my_string[149]
  out[14] = my_string[14]
  out[150] = my_string[150]
  out[151] = my_string[151]
  out[157] = my_string[157]
  out[158] = my_string[158]
  out[159] = my_string[159]
  out[15] = my_string[15]
  out[165] = my_string[165]
  out[166] = my_string[166]
  out[167] = my_string[167]
  out[173] = my_string[173]
  out[174] = my_string[174]
  out[175] = my_string[175]
  out[181] = my_string[181]
  out[182] = my_string[182]
  out[183] = my_string[183]
  out[189] = my_string[189]
  out[190] = my_string[190]
  out[191] = my_string[191]
  out[197] = my_string[197]
  out[198] = my_string[198]
  out[199] = my_string[199]
  out[205] = my_string[205]
  out[206] = my_string[206]
  out[207] = my_string[207]
  out[213] = my_string[213]
  out[214] = my_string[214]
  out[215] = my_string[215]
  out[21] = my_string[21]
  out[221] = my_string[221]
  out[222] = my_string[222]
  out[223] = my_string[223]
  out[229] = my_string[229]
  out[22] = my_string[22]
  out[230] = my_string[230]
  out[231] = my_string[231]
  out[237] = my_string[237]
  out[238] = my_string[238]
  out[239] = my_string[239]
  out[23] = my_string[23]
  out[245] = my_string[245]
  out[246] = my_string[246]
  out[247] = my_string[247]
  out[253] = my_string[253]
  out[254] = my_string[254]
  out[255] = my_string[255]
  out[29] = my_string[29]
  out[30] = my_string[30]
  out[31] = my_string[31]
  out[37] = my_string[37]
  out[38] = my_string[38]
  out[39] = my_string[39]
  out[45] = my_string[45]
  out[46] = my_string[46]
  out[47] = my_string[47]
  out[53] = my_string[53]
  out[54] = my_string[54]
  out[55] = my_string[55]
  out[5] = my_string[5]
  out[61] = my_string[61]
  out[62] = my_string[62]
  out[63] = my_string[63]
  out[69] = my_string[69]
  out[6] = my_string[6]
  out[70] = my_string[70]
  out[71] = my_string[71]
  out[77] = my_string[77]
  out[78] = my_string[78]
  out[79] = my_string[79]
  out[7] = my_string[7]
  out[85] = my_string[85]
  out[86] = my_string[86]
  out[87] = my_string[87]
  out[93] = my_string[93]
  out[94] = my_string[94]
  out[95] = my_string[95]
  return out
