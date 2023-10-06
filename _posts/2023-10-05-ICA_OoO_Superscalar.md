---
layout: article
title: NP, NP-Complete Problems
tags: Computer_Architecture
aside:
  toc: true
---

This blog talks about Out-of-Order (OoO) Superscalar pipeline. We will give what hardware do in each stage.

<!--more-->

# Fetch (in program order)

Fetch multiple sequential instructions in parallel. 

# Decode (in program order)

In parallel, decode all of the instr’s just fetched. 

# Rename & Dispatch (in program order)

## Rename 

Rename the architected registers (ArchitectedRegFile, ARF) with physical
registers (PhysicalRegFile, PRF). 

### Map Table

For each architected register, we record the physical register that curretly represent this ARF. (In this class, we assume the initial map table have already assign each ARF $si with a PRF $pi.)

### Free List

Keep record each PRF state (used or not)


### How to do in "Rename" stage

* Use mapping table, replace the source archi register to physical register. 

* For dst register, pick up a available physical register to replace it.

* Record the over-written Physical Reg

* Update the free list 

Example

lw $t0, 0 ($s1)
addu $t0, $t0, $s2

mapping table 

| s1 | p1 |
| s2 | p2 |
| t0 | p3 |

## Dispatch

* Make renamed instr’s eligible for execution by dispatching them to the IQ (Instruction
Queue) and the ROB (ReOrder Buffer). 

* Loads and stores are dispatched as two (micro)instr’s – one to the IQ to compute the addr and one to LSQ (LoadStoreQueue) for the
memory operation.

# Issue (Out of Order)

When an instr in the IQ has all of its source data and the FU (Functional Unit) it needs
is free, it is issued for execution.

# Writeback (Out of Order)

Writeback (Out Of Order - OoO): When the dst value has been computed it is written back to the PRF, the IQ, ROB and LSQ are updated – the instr <b>completes</b> execution. 

Note: Stores DO NOT write to cache at this stage, and the ARF is NOT updated

# Commit (in program order)

Commit the instr in order, that means the oldest completed instr in ROB can be commit. (No matter how wide commit is, cannot skip over uncompleted instructions)

In Commit stage, we update ARF and store value to memory.