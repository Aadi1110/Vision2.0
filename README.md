<html>
<body>
<h1> Vision 2.0 </h1>

<h2> Problem Statement </h2>

<h3> Arena Description: </h3>

<ol>
1. There are 2 paths (inner and outer square) and there are 4 connecting paths of different colours joining them. <br>
2. Bot can change from outer path to inner path or vice versa. Bot is allowed to move in <b> clockwise direction </b> only. The portion of    the arena in <b> black colour </b> is restricted for the movement of the bot. <br>
3. There will be 3 shapes (square, circle and triangle) of 2 different colours, distinguishing each block in 6 different ways. <br>
4. On the outermost path there will be 4 arrows at the end of connecting paths pointing in clockwise direction. These arrows mark the <b> Starting Zone </b> where the bot will be placed initially on any one of the arrows. <br>
5. The Centre of the arena is the home zone. <br>
6. The bot has to traverse the arena, complete a full round and finish at the home zone. <br>
</ol>

<center>
<img src = "k.png" alt = "Arena" width = "400" height = "400">
</center>

<h3> Task To-do: </h3>

<ol>
1. The bot is placed at one of the Starting Zones. <br>
2. An abbreviation which associate to specific colour and shape. <br>
  <ul>
   - <i> RT </i> for Red Triangle. <br>
   - <i> RS </i> for Red Square. <br>
   - <i> RC </i> for Red Circle. <br>
   - <i> YT </i> for Yellow Triangle. <br>
   - <i> YS </i> for Yellow Square. <br>
   - <i> YC </i> for Yellow Circle. <br>
 </ul>
3. The bot must then find the closest block which it can reach following a clockwise path. <br>
4. Signal must be sent to when bot stops moving. <br>
5. As soon as the bot stops moving, bot has to ask for input using the function provided. <br>
6. This continues till the bot has completed a full round around the center, then it should move to home via the connecting paths that it started on. <br>
7. On reaching home the bot should signal that it has finished the task. <br>
</ol>
<br>
<br>
<h2> The Approach </h2>
<b> Computer Vision </b> to process images, <b> Breadth-first search </b> to track the path and <b> Pybullet </b> to simulate the bot, are the major things used to complete the task. <br> 
First, a graph is created, in which edges are added in the direction of allowed movement. <br>
Aruco --- <br>
Then Breadth-first search is used to determine the path from the current position to the next destination. <br>
 
</body>
</html>




