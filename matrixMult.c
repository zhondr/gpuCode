#include <stdio.h>
#include <time.h>
 
int main()
{
  int m, n, p, q, c, d, k, sum;
  int first[10][10], second[10][10], multiply[10][10];
  double time_spent;

m=3;n=3;p=3;q=3;
/*  printf("Enter number of rows and columns of first matrix\n");
  scanf("%d%d", &m, &n);
  printf("Enter elements of first matrix\n");
*/ 
  for (c = 0; c < m; c++)
    for (d = 0; d < n; d++)
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
    for (c = 0; c < p; c++)
      for (d = 0; d < q; d++)
	second[c][d]=c+1;
//        scanf("%d", &second[c][d]); 
    
   for (int i=0;i<10000;i++) {
    sum=0;
    clock_t begin = clock();
    for (c = 0; c < m; c++) {
      for (d = 0; d < q; d++) {
        for (k = 0; k < p; k++) {
          sum = sum + first[c][k]*second[k][d];
        }
 
        multiply[c][d] = sum;
        sum = 0;
      }
    }
    clock_t end = clock();
    time_spent = time_spent + ((double)(end - begin) / CLOCKS_PER_SEC);
   }

    printf("Overall time: %f\n",time_spent);
    printf("Average time: %f\n",time_spent/10000);
    printf("Product of the matrices:\n");
 
    for (c = 0; c < m; c++) {
      for (d = 0; d < q; d++)
        printf("%d\t", multiply[c][d]);
 
      printf("\n");
    }
  
 
  return 0;
}

