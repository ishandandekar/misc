### Notes regarding regex
- Type the word itself to get exact matching
- **`.`** allows selecting any character, including special characters and spaces.
- Write alternative letters in **`[]`**. For example:
    > bar, ber, bir, bor, bur -> b\[aeiou\]r
- Write characters to omit after **`^`**
    > bar, ber, bir, bor, bur -> b\[^aeiou\]r
- Letter range can be defined as: **`[a-z]`**. This is case sensitive.
- Number range can be defined similarly: **`[0-9]`**
- Special characters such as **`+`**, **`*`** and **`?`**. These are used to specify how many time a character will be repeated.
    > br, ber, beer -> be*r
- **`+`** is used to specify that character will occur one-or-more times.
- To specify that a letter is optional, add **`?`** after it.
- Specify the occurence of a letter in **`{}`**. Add a **`,`** to specify that it should occur atleast that many times. Add limits after comma to specify frequency of character.
    > be{2}r -> beer  
    > be{3,}r -> beeer, beeeer  
    > be{1,3}r-> ber, beer, beeer
- Group text using **`()`** to reference or enforce rules.
    > ha-ha,haa-haa -> (ha)-\1,(haa)-\2