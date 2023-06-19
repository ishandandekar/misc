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