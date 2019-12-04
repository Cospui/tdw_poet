# POET with RAY

To run the code:

```
python run.py ./tdw_logs/log_0 --envs target cube_stack_target --batch_size 2 --repro_threshold -100 --mc_lower -99 --mc_upper 99
```

pass word:

```
Vyxqk2pQ
```

第一步：重新过一遍代码，看看对不对

第二部：加入并行计算



参数：simulate调用时候和里面的，```max_episode_length```以及```num_episode```。



问题：遗传算法在两者的reward都是0的时候无法优化。在复杂的环境下无法保证reward不是0，所以需要重新设计一个reward。

问题：```x,y,z```差别太小了，应该×一个倍数，现在是20，感觉有点小

