function CantileverRodStretchBendCompare
% Static Cosserat-rod comparison case matched to Stretching_bending defaults.
%
% The active Python model relaxes to a static equilibrium under nodal loads.
% For a clean comparison, this script solves the corresponding static
% Cosserat boundary-value problem by shooting. Axial stretching and bending
% use the same EA and B values as the Python model. Shear and twist are kept
% as very stiff penalty modes because the Cosserat constitutive matrices must
% remain invertible.

% Parameters matched to Stretching_bending/run_cantilever_demo.py defaults.
num_links = 30;
L = 1.0;
N = num_links + 1;
EA = 1000.0;
B = 0.5;
node_mass = 0.1;

% User-model coordinates: x is the straight rod axis, z is transverse.
force_user = [2.0; 0.0; 2.0];
gravity_user = 0.0;

% Cosserat coordinates follow the paper code: z is the straight rod axis,
% x is the plotted transverse direction. Map user [x;y;z] -> CR [z;y;x].
F_tip = [force_user(3); force_user(2); force_user(1)];
M_tip = [0; 0; 0];
g = [-gravity_user; 0; 0];

% Penalty stiffnesses for degrees of freedom absent in the no-twist DER model.
shear_penalty = 1e2;
twist_penalty = 1e2;
Kshear = shear_penalty * EA;
Ktwist = twist_penalty * B;

vstar = [0; 0; 1];
ds = L / (N - 1);

% Approximate the Python nodal masses as a distributed line mass.
rhoA = node_mass / ds;
rhoAg = rhoA * g;

Kse = diag([Kshear, Kshear, EA]);
Kbt = diag([B, B, Ktwist]);
Kse_inv = inv(Kse);
Kbt_inv = inv(Kbt);
Kse_vstar = Kse * vstar;

% Boundary conditions at the clamped root.
p0 = [0; 0; 0];
h0 = [1; 0; 0; 0];

% y := [p; h; n; m]
y = [zeros(2, N); linspace(0, L, N); zeros(10, N)];

% A straight-beam initial guess improves fsolve robustness.
root_moment_guess = cross([0; 0; L], F_tip);
G_guess = [F_tip; root_moment_guess];
load_scale = 0.0;
options = optimoptions( ...
    'fsolve', ...
    'Display', 'off', ...
    'MaxFunctionEvaluations', 5000, ...
    'MaxIterations', 1000, ...
    'FunctionTolerance', 1e-10, ...
    'StepTolerance', 1e-10);

load_steps = 20;
G_solution = G_guess;
for load_index = 1:load_steps
    load_scale = load_index / load_steps;
    G_solution = fsolve(@getResidual, G_solution, options);
end
load_scale = 1.0;
getResidual(G_solution);
visualize();

positions_user_axes = [y(3, :); y(2, :); y(1, :)]';
tip_displacement = positions_user_axes(end, :) - [L, 0, 0];
fprintf('Tip displacement in user axes [ux uy uz] = [% .8e % .8e % .8e] m\n', ...
    tip_displacement(1), tip_displacement(2), tip_displacement(3));
fprintf('Tip displacement magnitude = %.8e m\n', norm(tip_displacement));

    function visualize()
        plot(y(3, :), y(1, :), '-o');
        title('Static Stretching + Bending Cosserat Rod Comparison');
        xlabel('axial position, user x / CR z (m)');
        ylabel('transverse position, user z / CR x (m)');
        axis equal;
        grid on;
        drawnow;
    end

    function residual = getResidual(G_input)
        n0 = G_input(1:3);
        m0 = G_input(4:6);
        y(:, 1) = [p0; h0; n0; m0];

        for j = 1:N-1
            yj = y(:, j);
            k1 = ODE(yj);
            k2 = ODE(yj + k1*ds/2);
            k3 = ODE(yj + k2*ds/2);
            k4 = ODE(yj + k3*ds);
            y(:, j+1) = yj + ds * (k1 + 2*(k2 + k3) + k4) / 6;
        end

        nL = y(8:10, N);
        mL = y(11:13, N);
        residual = [load_scale * F_tip - nL; M_tip - mL];
    end

    function ys = ODE(y_in)
        h = y_in(4:7);
        n = y_in(8:10);
        m = y_in(11:13);

        h1 = h(1);
        h2 = h(2);
        h3 = h(3);
        h4 = h(4);

        R = eye(3) + 2/(h' * h) * ...
            [-h3^2 - h4^2, h2*h3 - h4*h1, h2*h4 + h3*h1;
              h2*h3 + h4*h1, -h2^2 - h4^2, h3*h4 - h2*h1;
              h2*h4 - h3*h1, h3*h4 + h2*h1, -h2^2 - h3^2];

        v = Kse_inv * (R' * n + Kse_vstar);
        u = Kbt_inv * (R' * m);

        ps = R * v;
        ns = -rhoAg;
        ms = -cross(ps, n);

        hs = [0,    -u(1), -u(2), -u(3);
              u(1),  0,     u(3), -u(2);
              u(2), -u(3),  0,     u(1);
              u(3),  u(2), -u(1),  0] * h / 2;

        ys = [ps; hs; ns; ms];
    end
end
