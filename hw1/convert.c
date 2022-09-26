#include <stdio.h>
#include <stdlib.h>
#include <string.h>

void fallback_print_usage() {
  printf("Usage: ./convert [int|long|float|double] number\n");
  printf("Example: ./convert float 3.14\n");
  exit(0);
}

void print_int(int x) {
  // one character per one bit
  char output[32 + 1] = {
      0,
  };

  /* YOUR CODE START HERE */
  int negative = 0;
  if(x < 0){
    x = -x;
    negative = 1;
  }

  int j = 31;

  while(1){
    if(x != 1){
      output[j] = (x % 2) + '0';
      x = x / 2;
      j--;
    }
    else if(x == 1){
      output[j] = 1 + '0';
      j--;
      break;
    }
  }

  int i = 0;
  for(i = j; i > -1; i--){
    output[i] = 0 + '0';
  }

  //2's complete
  if(negative == 1){
    //flip
    for(int i = 0; i < 32; i++){
      output[i] = (output[i] == '0')? '1': '0';
    }
    //+1
    for(int i = 31; i > -1; i--){
      if(output[i] == '1'){
        output[i] = '0';
      }else if(output[i] == '0'){
        output[i] = '1';
        break;
      }
    }

  }
  /* YOUR CODE END HERE */
  
  printf("%s\n", output);
}

void print_long(long x) {
  // one character per one bit
  char output[64 + 1] = {
      0,
  };

  /* YOUR CODE START HERE */
  int negative = 0;
  if(x < 0){
    x = -x;
    negative = 1;
  }

  int j = 63;

  while(1){
    if(x != 1){
      output[j] = (x % 2) + '0';
      x = x / 2;
      j--;
    }
    else if(x == 1){
      output[j] = 1 + '0';
      j--;
      break;
    }
  }

  int i = 0;
  for(i = j; i > -1; i--){
    output[i] = 0 + '0';
  }

  //2's complete
  if(negative == 1){
    //flip
    for(int i = 0; i < 64; i++){
      output[i] = (output[i] == '0')? '1': '0';
    }
    //+1
    for(int i = 63; i > -1; i--){
      if(output[i] == '1'){
        output[i] = '0';
      }else if(output[i] == '0'){
        output[i] = '1';
        break;
      }
    }

  }
  /* YOUR CODE END HERE */

  printf("%s\n", output);
}

void print_float(float x) {
  // one character per one bit
  char output[32 + 1] = {
      0,
  };

  /* YOUR CODE START HERE */
  // detect negative
  int negative = 0;
  if(x < 0){
    x = -x;
    negative = 1;
    output[0] = '1';
  }else{
    output[0] = '0';
  }

  // split into integer part and decimal part
  int integer_part = (int)x;
  float decimal_part = x - integer_part;

  // convert integer part
  char integer_list[32 + 1] = {
      0,
  };

  int j = 0;

  while(1){
    if(integer_part != 1){
      integer_list[j] = (integer_part % 2) + '0';
      integer_part = integer_part / 2;
      j++;
    }
    else if(integer_part == 1){
      integer_list[j] = 1 + '0';
      j++;
      break;
    }
  }

  // reverse
  int num = 0;
  num = strlen(integer_list);
  for(int i = 0; i > num/2; i++){
    char temp = integer_list[i];
    integer_list[i] = integer_list[num - 1 - i];
    integer_list[num - 1 - i] = temp;
  }

  // convert decimal part
  char decimal_list[32 + 1] = {
    0,
  };

  j = 0;

  while(1){
    decimal_part = decimal_part * 2;
    decimal_list[j] = (int)(decimal_part) + '0';
    decimal_part = decimal_part - (int)(decimal_part);
    j++;
    if(decimal_part == 0){
      if(j < 22) decimal_list[j] = 0 + '0';
      break;
    }
  }

  // exp + 127 -> exp part
  int n = strlen(integer_list);
  int exp = n - 1;
  exp = exp + 127;

  j = 8;
  while(1){
    if(exp != 1){
      output[j] = (exp % 2) + '0';
      exp = exp / 2;
      j--;
    }
    else if(exp == 1){
      output[j] = 1 + '0';
      j--;
      break;
    }
  }

  // mentissa +0..0 -> m part
  int integer_list_size = strlen(integer_list);
  int decimal_list_size = strlen(decimal_list);
  int length = integer_list_size + decimal_list_size;
  char mentissa_list[length - 1];

  char *integer = integer_list+1;
  strcpy(mentissa_list, integer);
  strcat(mentissa_list, decimal_list);
  
  int i = 0;
  for(i = length - 1; i < 23; i++){
    mentissa_list[i] = 0 + '0';
  }

  // s:1 + e:8 + m:23
  strcat(output, mentissa_list);
  /* YOUR CODE END HERE */

  printf("%s\n", output);
}

void print_double(double x) {
  // one character per one bit
  char output[64 + 1] = {
      0,
  };

  /* YOUR CODE START HERE */
  // 1. detect negative
  int negative = 0;
  if(x < 0){
    x = -x;
    negative = 1;
    output[0] = '1';
  }else{
    output[0] = '0';
  }

  // 2. split into integer part and decimal part
  int integer_part = (int)x;
  double decimal_part = x - integer_part;

  // 3. convert integer part
  char integer_list[64 + 1] = {
      0,
  };

  int j = 0;

  while(1){
    if(integer_part != 1){
      integer_list[j] = (integer_part % 2) + '0';
      integer_part = integer_part / 2;
      j++;
    }
    else if(integer_part == 1){
      integer_list[j] = 1 + '0';
      j++;
      break;
    }
  }

  // 4. reverse
  int num = 0;
  num = strlen(integer_list);
  for(int i = 0; i > num/2; i++){
    char temp = integer_list[i];
    integer_list[i] = integer_list[num - 1 - i];
    integer_list[num - 1 - i] = temp;
  }

  // 5. convert decimal part
  char decimal_list[64 + 1] = {
    0,
  };

  j = 0;

  while(1){
    decimal_part = decimal_part * 2;
    decimal_list[j] = (int)(decimal_part) + '0';
    decimal_part = decimal_part - (int)(decimal_part);
    j++;
    if(decimal_part == 0){
      if(j < 51) decimal_list[j] = 0 + '0';
      break;
    }
  }

  // 6. exp + 1023 -> exp part
  int n = strlen(integer_list);
  int exp = n - 1;
  exp = exp + 1023;

  j = 11;
  while(1){
    if(exp != 1){
      output[j] = (exp % 2) + '0';
      exp = exp / 2;
      j--;
    }
    else if(exp == 1){
      output[j] = 1 + '0';
      j--;
      break;
    }
  }

  // 7. mentissa +0..0 -> m part
  int integer_list_size = strlen(integer_list);
  int decimal_list_size = strlen(decimal_list);
  int length = integer_list_size + decimal_list_size;
  char mentissa_list[length];

  char *integer = integer_list+1;
  strcpy(mentissa_list, integer);
  strcat(mentissa_list, decimal_list);

  int i = 0;
  for(i = length - 1; i < 52; i++){
    mentissa_list[i] = 0 + '0';
  }

  // 8. s:1 + e:11 + m:52
  strcat(output, mentissa_list);
  /* YOUR CODE END HERE */

  printf("%s\n", output);
}

int main(int argc, char **argv) {
  if (argc != 3)
    fallback_print_usage();
  if (strcmp(argv[1], "int") == 0) {
    print_int(atoi(argv[2]));
  } else if (strcmp(argv[1], "long") == 0) {
    print_long(atol(argv[2]));
  } else if (strcmp(argv[1], "float") == 0) {
    print_float(atof(argv[2]));
  } else if (strcmp(argv[1], "double") == 0) {
    print_double(atof(argv[2]));
  } else {
    fallback_print_usage();
  }
}