#include <stdio.h>
#include <string.h>

#define MAX 50

typedef struct {
    int start, end, size;
    char process[10];
    int free;
} Block;

Block mem[MAX];
int n = 12;

/* Initial memory map */
void initialize()
{
    mem[0]  = (Block){0, 10, 10, "P5", 0};
    mem[1]  = (Block){10, 310, 300, "FREE", 1};
    mem[2]  = (Block){310, 400, 90, "P6", 0};
    mem[3]  = (Block){400, 1000, 600, "FREE", 1};
    mem[4]  = (Block){1000, 1500, 500, "P1", 0};
    mem[5]  = (Block){1500, 1850, 350, "FREE", 1};
    mem[6]  = (Block){1850, 2000, 150, "P2", 0};
    mem[7]  = (Block){2000, 2200, 200, "FREE", 1};
    mem[8]  = (Block){2200, 2300, 100, "P4", 0};
    mem[9]  = (Block){2300, 3150, 850, "FREE", 1};
    mem[10] = (Block){3150, 3500, 350, "P3", 0};
    mem[11] = (Block){3500, 4000, 500, "FREE", 1};

    n = 12;
}

/* Display memory */
void display()
{
    int i;

    printf("\n--------------------------------------------\n");
    printf(" Start\tEnd\tSize\tProcess\n");
    printf("--------------------------------------------\n");

    for (i = 0; i < n; i++)
    {
        printf(" %d\t%d\t%d KB\t%s\n",
               mem[i].start,
               mem[i].end,
               mem[i].size,
               mem[i].process);
    }

    printf("--------------------------------------------\n");
}

/* Insert block */
void insertBlock(int pos, Block b)
{
    int i;

    for (i = n; i > pos; i--)
        mem[i] = mem[i - 1];

    mem[pos] = b;
    n++;
}

/* Delete block */
void deleteBlock(int pos)
{
    int i;

    for (i = pos; i < n - 1; i++)
        mem[i] = mem[i + 1];

    n--;
}

/* Allocate memory */
void allocate(char process[], int size, int strategy)
{
    int i, index = -1;

    /* First Fit */
    if (strategy == 1)
    {
        for (i = 0; i < n; i++)
        {
            if (mem[i].free && mem[i].size >= size)
            {
                index = i;
                break;
            }
        }
    }

    /* Best Fit */
    else if (strategy == 2)
    {
        for (i = 0; i < n; i++)
        {
            if (mem[i].free && mem[i].size >= size)
            {
                if (index == -1 ||
                    mem[i].size < mem[index].size)
                    index = i;
            }
        }
    }

    /* Worst Fit */
    else if (strategy == 3)
    {
        for (i = 0; i < n; i++)
        {
            if (mem[i].free && mem[i].size >= size)
            {
                if (index == -1 ||
                    mem[i].size > mem[index].size)
                    index = i;
            }
        }
    }

    if (index == -1)
    {
        printf("\nNo suitable free block found!\n");
        return;
    }

    /* Exact fit */
    if (mem[index].size == size)
    {
        strcpy(mem[index].process, process);
        mem[index].free = 0;
    }

    /* Split block */
    else
    {
        Block allocated, remaining;

        allocated.start = mem[index].start;
        allocated.end = allocated.start + size;
        allocated.size = size;
        allocated.free = 0;
        strcpy(allocated.process, process);

        remaining.start = allocated.end;
        remaining.end = mem[index].end;
        remaining.size = remaining.end - remaining.start;
        remaining.free = 1;
        strcpy(remaining.process, "FREE");

        mem[index] = allocated;
        insertBlock(index + 1, remaining);
    }

    printf("\n%s allocated %d KB successfully.\n",
           process, size);

    printf("Allocated from %d KB to %d KB.\n",
           mem[index].start,
           mem[index].end);
}

/* Merge adjacent free blocks */
void mergeFree()
{
    int i;

    for (i = 0; i < n - 1; i++)
    {
        if (mem[i].free && mem[i + 1].free)
        {
            mem[i].end = mem[i + 1].end;
            mem[i].size = mem[i].end - mem[i].start;

            deleteBlock(i + 1);
            i--;
        }
    }
}

/* Free process */
void release(char process[])
{
    int i;

    for (i = 0; i < n; i++)
    {
        if (!mem[i].free &&
            strcmp(mem[i].process, process) == 0)
        {
            mem[i].free = 1;
            strcpy(mem[i].process, "FREE");

            printf("\n%s memory released successfully.\n",
                   process);

            mergeFree();
            return;
        }
    }

    printf("\nProcess %s not found!\n", process);
}

/* Main */
int main()
{
    int choice;
    int strategy;
    int size;
    char process[10];

    initialize();

    while (1)
    {
        printf("\n====================================\n");
        printf("       MEMORY ALLOCATION SYSTEM\n");
        printf("====================================\n");
        printf("1. Request Memory\n");
        printf("2. Free Memory\n");
        printf("3. Display Memory\n");
        printf("0. Exit\n");

        printf("\nEnter your choice: ");
        scanf("%d", &choice);

        switch (choice)
        {
            /* REQUEST */
            case 1:

                printf("\nEnter process name: ");
                scanf("%s", process);

                printf("Enter required memory (KB): ");
                scanf("%d", &size);

                if (size <= 0)
                {
                    printf("Invalid memory size!\n");
                    break;
                }

                printf("\nSelect Allocation Strategy\n");
                printf("1. First Fit\n");
                printf("2. Best Fit\n");
                printf("3. Worst Fit\n");

                printf("Enter choice: ");
                scanf("%d", &strategy);

                if (strategy < 1 || strategy > 3)
                {
                    printf("Invalid strategy!\n");
                    break;
                }

                allocate(process, size, strategy);

                break;

            /* FREE */
            case 2:

                printf("\nEnter process to free: ");
                scanf("%s", process);

                release(process);

                break;

            /* DISPLAY */
            case 3:

                display();

                break;

            /* EXIT */
            case 0:

                printf("\nProgram terminated.\n");
                return 0;

            default:

                printf("\nInvalid choice!\n");
        }
    }
}
