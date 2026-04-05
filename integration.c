#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <stddef.h>
#include <mpi.h>
#include <omp.h>
//set the safety stop for the divison to 50
#define MAX_DEPTH 50
#define OMP_TASK_DEPTH 4
// Define tags for mode 2
#define TAG_READY    100
#define TAG_TASK     101
#define TAG_NEW_TASK 102
#define TAG_STOP     103
#define TAG_REPORT   104

typedef struct {
    double a;
    double b;
    double tol;
    int depth;
} Task;

typedef struct {
    double sum;
    long long accepted;
    double active_time;
} Report;

typedef struct {
    Task *data;
    int size;
    int capacity;
} TaskStack;

//Functions to manage task stack in mode 2

void init_task_stack(TaskStack *stack, int initial_capacity) {
    stack->size = 0;
    stack->capacity = (initial_capacity > 0) ? initial_capacity : 16;
    stack->data = (Task *)malloc(stack->capacity * sizeof(Task));

    if (stack->data == NULL) {
        fprintf(stderr, "Failed to allocate task stack.\n");
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
}

void free_task_stack(TaskStack *stack) {
    free(stack->data);
    stack->data = NULL;
    stack->size = 0;
    stack->capacity = 0;
}

int task_stack_empty(TaskStack *stack) {
    return (stack->size == 0);
}

void push_task(TaskStack *stack, Task task) {
    if (stack->size == stack->capacity) {
        stack->capacity *= 2;
        Task *new_data = (Task *)realloc(stack->data, stack->capacity * sizeof(Task));
        if (new_data == NULL) {
            fprintf(stderr, "Failed to grow task stack.\n");
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
        stack->data = new_data;
    }

    stack->data[stack->size++] = task;
}

Task pop_task(TaskStack *stack) {
    if (stack->size == 0) {
        fprintf(stderr, "Tried to pop from an empty task stack.\n");
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    return stack->data[--stack->size];
}

//MPI data types to help
void create_mpi_types(MPI_Datatype *TASK_TYPE, MPI_Datatype *REPORT_TYPE) {
    {
        Task sample;
        int block_lengths[4] = {1, 1, 1, 1};
        MPI_Aint offsets[4];
        MPI_Datatype types[4] = {MPI_DOUBLE, MPI_DOUBLE, MPI_DOUBLE, MPI_INT};
        MPI_Aint base;

        MPI_Get_address(&sample, &base);
        MPI_Get_address(&sample.a, &offsets[0]);
        MPI_Get_address(&sample.b, &offsets[1]);
        MPI_Get_address(&sample.tol, &offsets[2]);
        MPI_Get_address(&sample.depth, &offsets[3]);

        for (int i = 0; i < 4; i++) {
            offsets[i] -= base;
        }

        MPI_Type_create_struct(4, block_lengths, offsets, types, TASK_TYPE);
        MPI_Type_commit(TASK_TYPE);
    }

    {
        Report sample;
        int block_lengths[3] = {1, 1, 1};
        MPI_Aint offsets[3];
        MPI_Datatype types[3] = {MPI_DOUBLE, MPI_LONG_LONG, MPI_DOUBLE};
        MPI_Aint base;

        MPI_Get_address(&sample, &base);
        MPI_Get_address(&sample.sum, &offsets[0]);
        MPI_Get_address(&sample.accepted, &offsets[1]);
        MPI_Get_address(&sample.active_time, &offsets[2]);

        for (int i = 0; i < 3; i++) {
            offsets[i] -= base;
        }

        MPI_Type_create_struct(3, block_lengths, offsets, types, REPORT_TYPE);
        MPI_Type_commit(REPORT_TYPE);
    }
}

// Function to determine wich function to use depending on the input
double f(double x, int func_id){
    switch (func_id) {
        case 0:
            return sin(x) + 0.5 * cos(3.0 * x);
        case 1: {
            double t = x - 0.3;
            return 1.0 / (1.0 + 100.0 * t * t);
        }
        case 2:
            return sin(200.0 * x) * exp(-x);
        default:
            fprintf(stderr, "Invalid func_id: %d (it must be 0, 1, or 2)\n", func_id);
            MPI_Abort(MPI_COMM_WORLD, 1);
            return 0.0;
    }
}


//choose k
static int choose_serial_K(int func_id) {
    if (func_id == 2) return 64;
    return 16;
}
static int choose_dynamic_K(int size) {
    int workers = size - 1;
    int K = 4 * workers;
    return (K > 0) ? K : 1;
}

static int choose_hybrid_K(int size, int func_id) {
    int workers = size;
    int base = 8 * workers;
    if (func_id == 2) base *= 2;
    return (base > 0) ? base : 1;
}
//simpson estimate function on [a,b]
double simpson_estimate(double a, double b, int func_id) {
    double m = 0.5 * (a + b);
    double fa = f(a, func_id);
    double fm = f(m, func_id);
    double fb = f(b, func_id);

    return (b - a) / 6.0 * (fa + 4.0 * fm + fb);
}

// Recursive adaptive simpson 

double adaptive_simpson(double a, double b, double tol, int func_id,long long *accepted_intervals, int depth) {

    //let m
    double m = 0.5 * (a + b);

    double s_whole = simpson_estimate(a, b, func_id);
    double s_left = simpson_estimate(a, m, func_id);
    double s_right = simpson_estimate(m, b, func_id);
    double s_refined = s_left + s_right;
    //calculate error
    double err = fabs(s_refined - s_whole);

    //check if we have reached to the safety stop
    if (depth >= MAX_DEPTH || err <= tol) {
        (*accepted_intervals)++;
        return s_refined;
    }

    return adaptive_simpson(a, m, tol / 2.0, func_id, accepted_intervals, depth + 1)
         + adaptive_simpson(m, b, tol / 2.0, func_id, accepted_intervals, depth + 1);
}

double run_serial(int func_id, double tol, long long *accepted_intervals) {
    int K = choose_serial_K(func_id);
    double width = 1.0 / (double)K;
    double result = 0.0;

    *accepted_intervals = 0;

    for (int i = 0; i < K; i++) {
        double a = i * width;
        double b = (i + 1) * width;

        result += adaptive_simpson(a, b, tol / (double)K,
                                   func_id, accepted_intervals, 0);
    }

    return result;
}

void dispatch_waiting_workers(TaskStack *pool,
                              int *waiting_workers, int *waiting_count,
                              int *worker_busy, int *active_workers,
                              int *worker_task_counts,
                              MPI_Datatype TASK_TYPE) {
    while (*waiting_count > 0 && !task_stack_empty(pool)) {
        int worker = waiting_workers[--(*waiting_count)];
        Task task = pop_task(pool);

        MPI_Send(&task, 1, TASK_TYPE, worker, TAG_TASK, MPI_COMM_WORLD);

        worker_busy[worker] = 1;
        (*active_workers)++;
        worker_task_counts[worker]++;
    }
}

//dynamic worker function 
void run_dynamic_worker(int func_id, MPI_Datatype TASK_TYPE, MPI_Datatype REPORT_TYPE) {
    Report local_report;
    local_report.sum = 0.0;
    local_report.accepted = 0;
    local_report.active_time = 0.0;

    TaskStack local_stack;
    init_task_stack(&local_stack, 16);

    //ask for the first task
    MPI_Send(NULL, 0, MPI_BYTE, 0, TAG_READY, MPI_COMM_WORLD);

    while (1) {
        MPI_Status status;
        MPI_Probe(0, MPI_ANY_TAG, MPI_COMM_WORLD, &status);

        if (status.MPI_TAG == TAG_STOP) {
            MPI_Recv(NULL, 0, MPI_BYTE, 0, TAG_STOP, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            MPI_Send(&local_report, 1, REPORT_TYPE, 0, TAG_REPORT, MPI_COMM_WORLD);
            break;
        }

        if (status.MPI_TAG == TAG_TASK) {
            Task task;
            MPI_Recv(&task, 1, TASK_TYPE, 0, TAG_TASK, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

            push_task(&local_stack, task);

            double phase_start = MPI_Wtime();

            while (!task_stack_empty(&local_stack)) {
                Task current = pop_task(&local_stack);

                double a = current.a;
                double b = current.b;
                double tol = current.tol;
                int depth = current.depth;

                double m = 0.5 * (a + b);

                double s_whole = simpson_estimate(a, b, func_id);
                double s_left = simpson_estimate(a, m, func_id);
                double s_right = simpson_estimate(m, b, func_id);
                double s_refined = s_left + s_right;

                double err = fabs(s_refined - s_whole);

                if (depth >= MAX_DEPTH || err <= tol) {
                    local_report.sum += s_refined;
                    local_report.accepted++;
                } else {
                    Task left_child;
                    left_child.a = a;
                    left_child.b = m;
                    left_child.tol = tol / 2.0;
                    left_child.depth = depth + 1;

                    Task right_child;
                    right_child.a = m;
                    right_child.b = b;
                    right_child.tol = tol / 2.0;
                    right_child.depth = depth + 1;

                    //Keep one half locally, return the other half to master
                    push_task(&local_stack, left_child);
                    MPI_Send(&right_child, 1, TASK_TYPE, 0, TAG_NEW_TASK, MPI_COMM_WORLD);
                }
            }

            local_report.active_time += MPI_Wtime() - phase_start;

            //request more work 
            MPI_Send(NULL, 0, MPI_BYTE, 0, TAG_READY, MPI_COMM_WORLD);
        }
    }

    free_task_stack(&local_stack);
}

//master dynamic

void run_dynamic_master(int size, int func_id, double tol, double start_time,
                        MPI_Datatype TASK_TYPE, MPI_Datatype REPORT_TYPE,
                        int *chosen_K, double *global_result,
                        long long *global_accepted, double *global_runtime,
                        Report *worker_reports, int *worker_task_counts) {
    int workers = size - 1;
    *chosen_K = choose_dynamic_K(size);

    *global_result = 0.0;
    *global_accepted = 0;
    *global_runtime = 0.0;

    for (int r = 0; r < size; r++) {
        worker_reports[r].sum = 0.0;
        worker_reports[r].accepted = 0;
        worker_reports[r].active_time = 0.0;
        worker_task_counts[r] = 0;
    }

    TaskStack pool;
    init_task_stack(&pool, *chosen_K + 8);

    double width = 1.0 / (double)(*chosen_K);

    for (int i = 0; i < *chosen_K; i++) {
        Task task;
        task.a = i * width;
        task.b = (i + 1) * width;
        task.tol = tol / (double)(*chosen_K);
        task.depth = 0;
        push_task(&pool, task);
    }

    int *worker_busy = (int *)calloc(size, sizeof(int));
    int *waiting_workers = (int *)malloc((workers > 0 ? workers : 1) * sizeof(int));
    int waiting_count = 0;
    int active_workers = 0;

    if (worker_busy == NULL || waiting_workers == NULL) {
        fprintf(stderr, "Failed to allocate master bookkeeping arrays.\n");
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    int stop_sent = 0;

    while (!stop_sent) {
        MPI_Status status;
        MPI_Probe(MPI_ANY_SOURCE, MPI_ANY_TAG, MPI_COMM_WORLD, &status);

        int source = status.MPI_SOURCE;
        int tag = status.MPI_TAG;

        if (tag == TAG_READY) {
            MPI_Recv(NULL, 0, MPI_BYTE, source, TAG_READY, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

            if (worker_busy[source]) {
                worker_busy[source] = 0;
                active_workers--;
            }

            if (!task_stack_empty(&pool)) {
                Task task = pop_task(&pool);
                MPI_Send(&task, 1, TASK_TYPE, source, TAG_TASK, MPI_COMM_WORLD);

                worker_busy[source] = 1;
                active_workers++;
                worker_task_counts[source]++;
            } else if (active_workers > 0) {
                waiting_workers[waiting_count++] = source;
            } else {
                //No queued work and no active worker: safe to stop 
                MPI_Send(NULL, 0, MPI_BYTE, source, TAG_STOP, MPI_COMM_WORLD);

                for (int i = 0; i < waiting_count; i++) {
                    MPI_Send(NULL, 0, MPI_BYTE, waiting_workers[i], TAG_STOP, MPI_COMM_WORLD);
                }

                stop_sent = 1;
            }
        } else if (tag == TAG_NEW_TASK) {
            Task returned_task;
            MPI_Recv(&returned_task, 1, TASK_TYPE, source, TAG_NEW_TASK, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

            push_task(&pool, returned_task);

            dispatch_waiting_workers(&pool,
                                     waiting_workers, &waiting_count,
                                     worker_busy, &active_workers,
                                     worker_task_counts,
                                     TASK_TYPE);
        }
    }

    //Collect final reports from all workers 
    for (int i = 0; i < workers; i++) {
        MPI_Status status;
        Report report;

        MPI_Recv(&report, 1, REPORT_TYPE, MPI_ANY_SOURCE, TAG_REPORT,
                 MPI_COMM_WORLD, &status);

        int source = status.MPI_SOURCE;
        worker_reports[source] = report;
        *global_result += report.sum;
        *global_accepted += report.accepted;
    }

    *global_runtime = MPI_Wtime() - start_time;

    free(worker_busy);
    free(waiting_workers);
    free_task_stack(&pool);
}

//Hybrid function implement static coarse interval distribution and local openMP computation
//No sending sub intervals to master
static void get_static_block(int total_items, int rank, int nranks,
                             int *start_index, int *count) {
    int base = total_items / nranks;
    int extra = total_items % nranks;

    if (rank < extra) {
        *count = base + 1;
        *start_index = rank * (*count);
    } else {
        *count = base;
        *start_index = extra * (base + 1) + (rank - extra) * base;
    }
}

double adaptive_simpson_hybrid(double a, double b, double tol, int func_id,
                               long long *accepted_intervals, int depth) {
    double m = 0.5 * (a + b);

    double s_whole = simpson_estimate(a, b, func_id);
    double s_left  = simpson_estimate(a, m, func_id);
    double s_right = simpson_estimate(m, b, func_id);
    double s_refined = s_left + s_right;

    double err = fabs(s_refined - s_whole);

    if (depth >= MAX_DEPTH || err <= tol) {
        #pragma omp atomic update
        (*accepted_intervals)++;
        return s_refined;
    }

    if (depth < OMP_TASK_DEPTH) {
        double left_sum = 0.0;
        double right_sum = 0.0;

        #pragma omp task shared(left_sum, accepted_intervals) firstprivate(a, m, tol, func_id, depth)
        {
            left_sum = adaptive_simpson_hybrid(a, m, tol / 2.0, func_id,
                                               accepted_intervals, depth + 1);
        }

        #pragma omp task shared(right_sum, accepted_intervals) firstprivate(m, b, tol, func_id, depth)
        {
            right_sum = adaptive_simpson_hybrid(m, b, tol / 2.0, func_id,
                                                accepted_intervals, depth + 1);
        }

        #pragma omp taskwait
        return left_sum + right_sum;
    } else {
        return adaptive_simpson_hybrid(a, m, tol / 2.0, func_id,
                                       accepted_intervals, depth + 1)
             + adaptive_simpson_hybrid(m, b, tol / 2.0, func_id,
                                       accepted_intervals, depth + 1);
    }
}
void run_hybrid_worker(int rank, int size, int func_id, double tol,
                       int chosen_K,
                       double *local_sum, long long *local_accepted) {
    int start_idx, count;
    double width = 1.0 / (double)chosen_K;

    *local_sum = 0.0;
    *local_accepted = 0;

    get_static_block(chosen_K, rank, size, &start_idx, &count);

    #pragma omp parallel
    {
        #pragma omp single
        {
            for (int i = start_idx; i < start_idx + count; i++) {
                double a = i * width;
                double b = (i + 1) * width;
                double local_tol = tol / (double)chosen_K;

                #pragma omp task shared(local_sum, local_accepted) firstprivate(a, b, local_tol, func_id)
                {
                    double partial = adaptive_simpson_hybrid(a, b, local_tol,
                                                             func_id, local_accepted, 0);

                    #pragma omp atomic update
                    *local_sum += partial;
                }
            }
        }
    }
}
void run_hybrid_master(int rank, int size, int func_id, double tol) {
    int chosen_K = choose_hybrid_K(size, func_id);

    double local_sum = 0.0;
    double global_sum = 0.0;

    long long local_accepted = 0;
    long long global_accepted = 0;

    MPI_Barrier(MPI_COMM_WORLD);
    double t0 = MPI_Wtime();

    run_hybrid_worker(rank, size, func_id, tol, chosen_K, &local_sum, &local_accepted);

    MPI_Reduce(&local_sum, &global_sum, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
    MPI_Reduce(&local_accepted, &global_accepted, 1, MPI_LONG_LONG, MPI_SUM, 0, MPI_COMM_WORLD);

    double t1 = MPI_Wtime();

    if (rank == 0) {
        printf("Mode: %d (Hybrid MPI + OpenMP)\n", 2);
        printf("Processes: %d\n", size);
        printf("Threads per process: %d\n", omp_get_max_threads());
        printf("Function ID: %d\n", func_id);
        printf("Tolerance: %.12e\n", tol);
        printf("Chosen hybrid K: %d\n", chosen_K);
        printf("Integral estimate: %.15f\n", global_sum);
        printf("Accepted intervals: %lld\n", global_accepted);
        printf("Runtime in seconds: %.6f\n", t1 - t0);
    }
}
int main(int argc, char *argv[]) {
    int rank, size;
    MPI_Datatype TASK_TYPE, REPORT_TYPE;

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    create_mpi_types(&TASK_TYPE, &REPORT_TYPE);

    if (argc != 4) {
        if (rank == 0) {
            fprintf(stderr,
                    "Command should be: mpirun -np P ./integration func_id mode tol\n"
                    "Examples:\n"
                    "  mpirun -np 1 ./integration 1 0 1e-8\n"
                    "  mpirun -np 4 ./integration 1 1 1e-8\n"
                    "  OMP_NUM_THREADS=4 mpirun -np 4 ./integration 1 2 1e-8\n");
        }
        MPI_Type_free(&TASK_TYPE);
        MPI_Type_free(&REPORT_TYPE);
        MPI_Finalize();
        return 1;
    }

    int func_id = atoi(argv[1]);
    int mode = atoi(argv[2]);
    double tol = atof(argv[3]);

    //check the tolerance if it's not negative number
    if (tol <= 0) {
        if (rank == 0) {
            fprintf(stderr, "Tolerance must be greater than 0\n");
        }
        MPI_Type_free(&TASK_TYPE);
        MPI_Type_free(&REPORT_TYPE);
        MPI_Finalize();
        return 1;
    }

    //chekc if the func_id is in the range of 0 to 2
    if (func_id < 0 || func_id > 2) {
        if (rank == 0) {
            fprintf(stderr, "func_id must be 0, 1, or 2\n");
        }
        MPI_Type_free(&TASK_TYPE);
        MPI_Type_free(&REPORT_TYPE);
        MPI_Finalize();
        return 1;
    }

    //chekc if the mode is in the range of 0 to 2
    if (mode < 0 || mode > 2) {
        if (rank == 0) {
            fprintf(stderr, "mode must be 0, 1, or 2\n");
        }
        MPI_Type_free(&TASK_TYPE);
        MPI_Type_free(&REPORT_TYPE);
        MPI_Finalize();
        return 1;
    }

    if (mode == 0) {
        if (rank == 0) {
            long long accepted_intervals = 0;

            double t0 = MPI_Wtime();
            double result = run_serial(func_id, tol, &accepted_intervals);
            double t1 = MPI_Wtime();

            printf("Mode: %d (Serial Baseline)\n", mode);
            printf("Processes: %d\n", size);
            printf("Function ID: %d\n", func_id);
            printf("Tolerance: %.12e\n", tol);
            printf("Initial serial K: %d\n", choose_serial_K(func_id));
            printf("Integral estimate: %.15f\n", result);
            printf("Accepted intervals: %lld\n", accepted_intervals);
            printf("Runtime in seconds: %.6f\n", t1 - t0);
        }
    }
    else if (mode == 1) {
        if (size < 2) {
            if (rank == 0) {
                fprintf(stderr, "Mode 1 needs at least 2 MPI processes.\n");
            }
            MPI_Type_free(&TASK_TYPE);
            MPI_Type_free(&REPORT_TYPE);
            MPI_Finalize();
            return 1;
        }

        MPI_Barrier(MPI_COMM_WORLD);
        double start_time = MPI_Wtime();

        if (rank == 0) {
            int chosen_K = 0;
            double global_result = 0.0;
            long long global_accepted = 0;
            double global_runtime = 0.0;

            Report *worker_reports = (Report *)malloc(size * sizeof(Report));
            int *worker_task_counts = (int *)malloc(size * sizeof(int));

            if (worker_reports == NULL || worker_task_counts == NULL) {
                fprintf(stderr, "Issue with memory allocation failed on rank 0.\n");
                MPI_Abort(MPI_COMM_WORLD, 1);
            }

            run_dynamic_master(size, func_id, tol, start_time,
                               TASK_TYPE, REPORT_TYPE,
                               &chosen_K, &global_result, &global_accepted,
                               &global_runtime, worker_reports, worker_task_counts);

            printf("Mode: %d (MPI Dynamic Master/Worker)\n", mode);
            printf("Processes: %d\n", size);
            printf("Function ID: %d\n", func_id);
            printf("Tolerance: %.12e\n", tol);
            printf("Chosen initial K: %d\n", chosen_K);
            printf("Integral estimate: %.15f\n", global_result);
            printf("Accepted intervals: %lld\n", global_accepted);
            printf("Runtime in seconds: %.6f\n", global_runtime);

            free(worker_reports);
            free(worker_task_counts);
        } else {
            run_dynamic_worker(func_id, TASK_TYPE, REPORT_TYPE);
        }
    }
    else if (mode == 2) {
        run_hybrid_master(rank, size, func_id, tol);
    }

    MPI_Type_free(&TASK_TYPE);
    MPI_Type_free(&REPORT_TYPE);
    MPI_Finalize();
    return 0;
}