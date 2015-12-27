#include <mpi.h>

#include <stdlib.h>
#include <unistd.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <sys/time.h>

#define _REENTRANT
#define _PRINT_TO_FILE

int main(int argc, char **argv) {

  int nodes = 0;
  double modelingTime = 0;
  double currentTime = 0;
  int sub_nodes = 0;
  double left_b = 0;
  double right_b = 0;
  MPI_Status status;

  int myrank, total;
  double *T_n1;        // step n and n+1 (sub mitrix)
  double *t_n, *t_n1;  // host processor
  double *t_n_transfer;

  struct timeval start, end;

  int i, j;
  int intBuf[2];
  int intTBuf[2];
  int up, down;

  double a_t;
  double deltaT;
  double deltaX;

  FILE *fp;

  nodes = 4096000;
  sub_nodes = 0;
  modelingTime = 0.012;
  //a_t = (210.0/(880.0*2700.0));
  a_t=2;
  deltaT = 0.0001;
  deltaX = 0.1;

  MPI_Init (&argc, &argv);
  MPI_Comm_size (MPI_COMM_WORLD, &total);
  MPI_Comm_rank (MPI_COMM_WORLD, &myrank);
  printf ("Total=%d, rank=%d\n", total, myrank);

  t_n =  (double *) malloc (sizeof(double)*nodes);
  t_n1 = (double *) malloc (sizeof(double)*nodes);

  if (!myrank) {
    for(i=0; i<nodes; i++) {
        t_n[i]  = 0;
        t_n1[i] = 0;
    }
  };

  printf("total: %d\n", total);
  printf("nodes: %d\n", nodes);

  if (!myrank) {
    intTBuf[0] = (int)(nodes/total);
    intTBuf[1] = 0;
  };

  MPI_Bcast((void *)intTBuf, 2, MPI_INT, 0, MPI_COMM_WORLD);

  sub_nodes = intTBuf[0];

  T_n1 = (double *) malloc (sizeof(double)*sub_nodes);
  for(i=0; i<sub_nodes; i++) {
    T_n1[i] = 0;
  }

#ifdef PRINT_TO_FILE
  if(!myrank) {
    fp = fopen("./animation.plot","w");
    fprintf(fp, "set xrange[0:%d]\n", nodes);
    fprintf(fp, "set yrange[-3:10]\n");
  }
#endif

//  printf("RANK: %d, sub_nodes: %d\n", myrank, sub_nodes);

  if(!myrank) {
    gettimeofday(&start,NULL);
  }

  MPI_Scatter((void *)t_n, sub_nodes, MPI_DOUBLE, (void *)T_n1, sub_nodes, MPI_DOUBLE, 0, MPI_COMM_WORLD);

  up = myrank+1;
  down = myrank-1;

  if(myrank == 0)
	down = MPI_PROC_NULL;
  if(myrank == (total-1))
	up = MPI_PROC_NULL;

  while(currentTime < modelingTime) {
    //printf("Time: %f\n", currentTime);

    //MPI_Bcast((void *)t_n,  nodes, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    right_b = T_n1[sub_nodes-1];
    left_b = T_n1[0];

    if ((myrank % 2) == 0) {
	//printf("RANK:%d HERE 1 sendTo:%d \n", myrank, up);
	MPI_Send(&T_n1[sub_nodes-1], 1, MPI_DOUBLE, up, 0, MPI_COMM_WORLD);
 	MPI_Recv(&right_b, 1, MPI_DOUBLE,  up, 1, MPI_COMM_WORLD, &status);
    }
    else {
	//printf("RANK:%d HERE 1 sendTo:%d \n", myrank, down);
	MPI_Recv(&left_b,  1, MPI_DOUBLE, down, 0, MPI_COMM_WORLD, &status);
        MPI_Send(&T_n1[0], 1, MPI_DOUBLE, down, 1, MPI_COMM_WORLD);
    }
    if ((myrank % 2) == 1) {
	//printf("RANK:%d HERE 2 sendTo:%d \n", myrank, up);
	MPI_Send(&T_n1[sub_nodes-1], 1, MPI_DOUBLE, up, 2, MPI_COMM_WORLD);
        MPI_Recv(&right_b,  1, MPI_DOUBLE, up, 3, MPI_COMM_WORLD, &status);
    }
    else {
	//printf("RANK:%d HERE 2 sendTo:%d \n", myrank, down);
	MPI_Recv(&left_b, 1, MPI_DOUBLE, down, 2, MPI_COMM_WORLD, &status);
        MPI_Send(&T_n1[0], 1,MPI_DOUBLE, down, 3, MPI_COMM_WORLD);
    }

    //printf("RANK: %d, left: %f, right: %f\n", myrank, left_b, right_b);

    for(i = 0; i<sub_nodes; i++) {
      if( (i + myrank*sub_nodes) == 0 ) { // GU 2
        //T_n1[i] = (1.)*deltaX/a_t + T_n1[i+1];
        T_n1[i] = 3;
        continue;
      }

      if( (i + myrank*sub_nodes) == (nodes-1) ) { // GU 1
        T_n1[i] = -1;
        continue;
      }

     if( (i == 0) ) {
         T_n1[i] = a_t*(left_b -
           2*T_n1[i] + T_n1[i + 1])*deltaT/(deltaX*deltaX) + T_n1[i];
        continue;
     }

     if( (i == sub_nodes-1) ) {
         T_n1[i] = a_t*(T_n1[i - 1] -
            2*T_n1[i] + right_b)*deltaT/(deltaX*deltaX) + T_n1[i];
        continue;
     }

     if((i > 0) && (i < sub_nodes-1)) {

     	T_n1[i] = a_t*(T_n1[i - 1] -
       	2*T_n1[i] + T_n1[i + 1])*deltaT/(deltaX*deltaX) + T_n1[i];

     	//printf("RANK %d is working %d from %d is %f\n", myrank, i, sub_nodes, T_n1[i]);
     }

#ifdef PRINT_TO_FILE
    MPI_Gather((void *)T_n1, sub_nodes, MPI_DOUBLE,
	       (void *)t_n1, sub_nodes, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    if(!myrank) {
      t_n = t_n1;
      fprintf(fp, "plot '-' u 1:2 w linespoints lw 1\n");
      for(j=0; j<nodes; j++) {
         fprintf(fp, "%d %f\n",j, t_n[j]);
      }
      fprintf(fp, "end\n");
      fprintf(fp, "pause 0.0001\n");
    }
#endif

    }
    currentTime += deltaT;
  }


if(!myrank) {
      gettimeofday(&end, NULL);
      printf("%ld\n", ((end.tv_sec * 1000000 + end.tv_usec)-(start.tv_sec * 1000000 + start.tv_usec)));
}

/*
  MPI_Gather((void *)T_n1, sub_nodes, MPI_DOUBLE,
	       (void *)t_n1, sub_nodes, MPI_DOUBLE, 0, MPI_COMM_WORLD);
  if (!myrank)  {
    printf("END:\n");
    for (i=0; i<nodes; i++)
      printf ("%g\n", t_n1[i]);
  }
*/
  printf("RANK: %d - EXIT\n", myrank);
  MPI_Finalize();

#ifdef PRINT_TO_FILE
  if(!myrank) {
    fclose(fp);
  }
#endif

  exit(0);
}
