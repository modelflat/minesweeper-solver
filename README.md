# Simple minesweeper solver

Simple minesweeper solver implemented in Python. Uses PyAutoGUI to get visual of 
minesweeper from screen and parses field state from it.

For actual decision making it uses 4 main stages:

* Random start - opens cells at random until enough space appear
* Simple calculation - digs/marks unambiguous cells
* Constraint analysis - makes more complicated decision by outlining and analysing constraints 
for empty cells
* Risking - if constraint analysis cannot tackle the problem, risks to open a cell with
least probability of having a mine

The last step can be replaced by something more sophisticated

Also, there are many inefficiencies here and there, whatever


![](https://raw.githubusercontent.com/modelflat/minesweeper-solver/master/img/minesweeper.gif)


