%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Algorithm:    Minimum Population Search
%
% Authors:      Antonio Bolufe-Rohler and Stephen Chen
% Description:  Minimum Population Search is a simple optimization
%               algorithm especially suited for multi-modal problems.
% Input Aguments:
%       FUN     is a string (objective) function name.
%       DIM     dimensionality of the objective function.
%       maxFEs  maximum number of function evaluations.
% Parameters:
%       alpha   represents the fraction of the main space diagonal used for the initial threshold value.
%       gamma   controls the decay rate of the threshold.
%       popsize population size. In standard implementation of MPS popsize = DIM.
%       bound   search space bounds. Symmetrical box constraints are assumed (as in BBOB, CEC and LSGO contests):
%               lower_bounds=-bound, upper_bounds=bound.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function [xmin fval] = MPS_bounded(FUN, popsize, maxFEs, lbounds, ubounds)
    %% MPS Parameters
    alpha = 0.3; % Original: 0.3
    beta = 2;  % Original: 2
    gamma = 3; % Original: 3
    DIM = addim;
    bound = 1;
    d = sqrt(DIM)*2*bound;  % Search Space Diagonal

    %% Bounds
    mid = (ubounds+lbounds)/2;
    range = (ubounds-lbounds)/2;

    %% Initial Population
    Population = zeros(2*popsize, DIM);
    f_Pop = Inf*ones(2*popsize,1);
    Population(1:popsize, :) =  (bound/2)*(((unifrnd(0,1,popsize,DIM)<0.5)*2)-1);

    %% Scaling solutions to actual bounds
    Trial = Population(1:popsize,:);
    Scaled = repmat(mid,popsize,1) + Trial .* repmat(range,popsize,1);

    %% Evaluating the initial population
    f_Pop(1:popsize,:) = feval(FUN, Scaled(1:popsize,:)');
    FEs = popsize;

    %% Main loop
    while  ( FEs < maxFEs)
        %% Obtaining current best individuals
        [~, indexes] = sort(f_Pop);
        pop = Population(indexes(1:popsize),:);

        %% Updating threshold
        min_step =  max(alpha*d* ((maxFEs-FEs)/maxFEs)^gamma, 1e-05);
        max_step = beta*min_step;

        %% Population Centroid
        centroid = repmat(sum(pop)/popsize, popsize, 1);

        %% Difference Vectors
        dif = normr(centroid - pop);

        %% Difference Vector Scaling Factor
        F = unifrnd(-max_step, max_step, popsize,1);

        %% Orthogonal Vectors
        orth = normr(normrnd(0,1,popsize,DIM));
        orth = normr(orth - repmat(dot(orth',dif')',1,DIM).*dif);

        %% Orthogonal Step Scaling Factor
        min_perp = sqrt(max(min_step^2-abs(F).^2,0));
        max_perp = sqrt(max(max_step^2-abs(F).^2,0));
        FO = unifrnd(min_perp, max_perp);

        %% New Solutions & Clamping
        Population(indexes(popsize+1:2*popsize),:) = max( min(pop + ...                 % Current Population
                                                            repmat(F,1,DIM).*dif + ...  % Difference Vector Step
                                                            repmat(FO,1,DIM).*orth,...  % Orthogonal Step
                                                         bound),...
                                                     -bound);

        %% Scaling solutions to actual bounds
        Trial = Population(indexes(popsize+1:2*popsize),:);
        Scaled = repmat(mid,popsize,1) + Trial .* repmat(range,popsize,1);

        %% Evaluation
        f_Pop(indexes(popsize+1:2*popsize)) = feval(FUN, Scaled');
        FEs = FEs + popsize;
    end

    %% Final Result
    [sorted, indexes] = sort(f_Pop);
    xmin = Population(indexes(1),:);

    xmin = mid + xmin .* range;

    fval = sorted(1);
end