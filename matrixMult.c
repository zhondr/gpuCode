#include <stdio.h>
#include <time.h>
 
int main()
{
  int M, N, c, d, k, sum;
  int first[100][100], second[100][100], multiply[100][100];
  double time_spent; 
  int cycleNum = 10000;

  N=100; 
/*  printf("Enter number of rows and columns of first matrix\n");
  scanf("%d%d", &m, &n);
  printf("Enter elements of first matrix\n");
*/ 
 for (M=3;M<=N;M++){ 
  for (c = 0; c < M; c++)
    for (d = 0; d < M; d++)
	first[c][d]=c+1;
	    //      scanf("%d", &first[c][d]);
 
/*  printf("Enter number of rows and columns of second matrix\n");
  scanf("%d%d", &p, &q);
 
  if (n != p)
    printf("The multiplication isn't possible.\n");
  else
  {
    printf("Enter elements of second matrix\n");
*/ 
    for (c = 0; c < M; c++)
      for (d = 0; d < M; d++)
	second[c][d]=c+1;
//        scanf("%d", &second[c][d]); 
    
   for (int i=0;i<cycleNum;i++) {
    sum=0;
    clock_t begin = clock();
    for (c = 0; c < M; c++) {
      for (d = 0; d < M; d++) {
        for (k = 0; k < M; k++) {
          sum = sum + first[c][k]*second[k][d];
        }
 
        multiply[c][d] = sum;
        sum = 0;
      }
    }
    clock_t end = clock();
    time_spent = time_spent + ((double)(end - begin) / CLOCKS_PER_SEC);
   }

   printf("%f\n",time_spent/cycleNum);
/*   
    printf("Overall time: %f\n",time_spent);
    printf("Average time: %f\n",time_spent/10000);
    printf("Product of the matrices:\n");
 
    for (c = 0; c < M; c++) {
      for (d = 0; d < M; d++)
        printf("%d\t", multiply[c][d]);
 
      printf("\n");
    }*/
  
 }
  return 0;
}

